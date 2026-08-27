# GDN2 Triton-Ascend NPU 适配与优化实验记录

环境：Atlas A2 (MIX_AIC)，CANN 9.0.0，torch_npu，Triton-Ascend (BiSheng codegen)，
conda env `syl-fla`。正确性门禁：`/tmp/gdn2_validate_full.py`（22 项，fwd/bwd/varlen/
多 dtype/非整块/initial_state），每次优化后全量复测。

---

## Phase 1 — NPU 移植与功能正确性

| 子阶段 | 内容 | 结果 |
|--------|------|------|
| 1a | `wy_fast.py`：`recompute_w_u_fwd_gdn2_npu` | 前向链路可用 |
| 1b | `chunk_intra.py`：fwd_intra（非 token-parallel）+ bwd_intra | fwd 门禁全绿 |
| 1c | `chunk_bwd.py`：`bwd_wy_dqkg_fused` | 反向链路可用 |
| 1d | `fused_recurrent.py` | 短序列/参考实现可用 |
| 1e | 全量门禁 + 修复 `_bwd_intra_g_row_stride` 缺 H 参数 bug | **22/22 PASS** |

Phase 1 结束时 bwd intra 的 `dq_db`/`dk_dg` 均为逐列标量 j-loop（KDA safe_gate
风格），正确但慢。

## Phase 2 — 性能优化迭代

profiling 方法：`.agents/skills/fla-ascend-performance/scripts/profile_npu.py
--metrics PipeUtilization` + `analyze_profile.py`（op_statistic / kernel_details）。
代表性 workload：B2T1024H4K64V64 bf16 fwd+bwd（NT=16, BC=32 → NC=2）。

### 2-a：dq_db 去标量化（首尝试，发现 BiSheng codegen 病态）

- **定位**：旧 `dq_db` 为 per-column 标量循环（`for i_k in range(BK)` 逐列累加），
  profile 显示 aic/aiv scalar_ratio ≈ 0.53/0.61，标量管线主导。
- **手段**：把对角块 j-loop 改写为 masked `tl.dot`（下三角掩码），指数用
  block-首行参考因子分解 `exp2(g_i - g_j) = exp2(g_i - g_ref) * exp2(g_ref - g_j)`。
- **结果（失败）**：kernel 7043us → 271002us（**38× 回归**）。aiv_scalar 0.464、
  aiv_mte2 0.471。
- **归因**：BiSheng 对「动态 bound 循环内 `tl.dot` + 循环外直线 `tl.dot`」的混合
  结构生成病态代码（寄存器/UB 排布坍塌）。这是编译器行为，不是算法问题。

### 2-b：统一循环架构 + UB 溢出修复

- **手段**：把 past 块与对角块合并进同一个动态循环 `for i_j in range(0, i_i+1)`，
  past 用 `m_use = (i_j < i_i) | m_i_diag` 掩码选择，所有 `tl.dot` 都在循环内 ——
  消除「循环内 dot + 直线 dot」混合结构。`dk_dg` 同手法。
- **UB 溢出**：fp32 B2T256H2K64V64 出现 `ub overflow, requires 2950144 bits while
  1572864 bits available`。实测 dq_db 峰值 ≈ 45 个 fp32 [BC,BK] tile（BK=64）。
  将 `_BWD_INTRA_DQ_MEM_MULT` 10.5 → 30.0，模型选 BK=32（1475584 bits，通过）。
- **结果**：`dk_dg` **2.16×** 加速（其循环天然退化为单次迭代 NC=1 可完全展开）；
  `dq_db` 因 BK 32 化 + 动态循环仍在，scalar_ratio 0.528/0.611 未根治。
  代价：`dkt_future` 共享 BK，64→32 导致任务数翻倍，739us → 1424us。

### 2-c：统一循环 NaN 修复（mid-row 参考因子分解）

- **定位**：门禁全红，bwd 梯度 NaN。逐 case 调试（`/tmp/gdn2_dbg_one.py`）发现
  对角三角内 `j > i`（列在行后）时 `g_ref(首行) - g_j` 可为正 → `exp2(正值)` 溢出
  inf，与掩码 0 相乘得 NaN。
- **手段**：参考行改为「块中行」（clamped）：`i_mid = i_ti + min(BC//2, T_cur-i_ti-1)`。
  两个因子 `exp2(g_mid - g_j)`、`exp2(g_i - g_mid)` 指数均被半块衰减上界约束，
  三角内无正指数溢出路径。越界行因子钳 0，避免 `inf * 0`。
- **结果**：**22/22 PASS**。

### 2-d：dq_db 拆分为 dq_past + dq_db_diag（本轮核心）

- **定位**：拆分前 profile（同 workload）：`dq_db` 占总时间 **22.1%**（最大单
  kernel），scalar_ratio 0.528/0.611；`dk_dg` 因单次迭代循环拿到 2.16×，证明
  「动态循环」本身是 scalar 开销主源。
- **手段**：仿照已有的 `dkt_future + dk_dg` 配对：
  - `dq_db_diag`（直线代码）：只算对角块，mid-row 参考，masked `tl.dot`；
    拥有 db partial 与 b-gated dk2 缩放；写 `own + dq`（dq 为上一阶段梯度）。
  - `dq_past`（动态循环）：只算 past 块（j < i_i），同样 mid-row 参考保持数值
    与 2-c 验证版一致；**累加模式**——读回 diag 已写的 dq2/dk2/dg2/db2 再加
    own 贡献（同流顺序执行，无需初始化缓冲）。
  - 1-based 编码 i_i（grid 索引 +1）使 sub-chunk 0（无 past）不发射任务。
  - BK 独立化：DQ_MEM_MULT 30.0 → 9.0，**BK 32 → 64**（K=64 时）；
    `dkt_future` 改用 BK_dk（mult 5.5），消除 2-b 引入的 739→1424us 回归。
- **坑**：Edit 工具对该文件失效（未 git 跟踪），改用带断言的 Python 补丁脚本
  （`/tmp/patch_phase2d*.py`）；首次补丁漏传 BK 参数（`dynamic_func() missing
  'BK'`），补回 common_launch。
- **结果**（B2T1024H4K64V64 bf16，kernel_details 实测）：

| kernel | 拆分前 | 拆分后 |
|--------|--------|--------|
| dq_db（统一循环） | >600us（占 22.1%） | — |
| dq_past | — | 45.7us |
| dq_db_diag | — | 105.2us |
| dkt_future | 739~1424us（BK 回归） | **39.1us** |
| dk_dg（对照） | 111.4us | 111.4us（不变 ✓） |

  bwd intra 四 kernel 合计 **301.4us**；`dq_past`+`dq_db_diag` = 150.9us，相对
  旧 dq_db 约 **4×**。门禁 **22/22 PASS**。
- **端到端基准**（synced timing，warmup 3 / repeat 10）：

| shape | fwd | fwd+bwd |
|-------|-----|---------|
| B2T256H2K64V64 fp32 | 0.9ms | 2.8ms |
| B2T1024H4K64V64 bf16 | 0.9ms | 3.0ms |
| B2T1024H4K128V128 bf16 | 1.1ms | 2.7ms |

> 口径勘误：profile schedule active=2，kernel_details 的 Duration 为两步聚合，
> 单步数值应减半（即拆分后 dq_past 22.9us / dq_db_diag 52.6us / dkt_future
> 19.6us / dk_dg 55.7us，bwd intra 合计 ~150us/步）。

## Phase 3 — fwd intra 路径切换（token_parallel → sub_chunk）

- **定位**：拆分后最大 GDN2 kernel 是 fwd 的 `intra_token_parallel`
  （单步 102.9us，vec 0.631 / MTE3 0.566）：逐 token 标量 j-loop，每步对
  [BH]=4 元素窄 store，海量碎片 store 把 MTE3 打满。
- **发现**：仓库内已有块式 `chunk_gdn2_fwd_kernel_intra_sub_chunk_npu`（直线
  `tl.dot` + mid-row 参考，无 BiSheng 混合结构风险），仅被 `safe_gate=False`
  默认路径旁路。
- **A/B 实验**（B2T1024H4K64V64 bf16）：safe_gate=False fwd 1.00ms vs
  safe_gate=True fwd 0.88ms；数值 o diff 6.1e-05（bf16 精度级），ht 完全一致。
- **手段**：`chunk_gdn2_fwd_intra_npu` 无条件走 sub_chunk kernel（safe_gate
  在 NPU 后端不再区分 intra 变体——两者数值等价）；token_parallel 文件保留
  为 backend API。
- **结果**：单步 `token_parallel` 102.9us → `sub_chunk` **78.7us（-24%）**；
  未改动 kernel 全部持平（dk_dg 55.7→57.7、dq_db_diag 52.6→56.1、dq_past
  22.9→27.2、dkt_future 19.6→23.6，波动范围内）。门禁 **22/22 PASS**。
  端到端 K64 fwd+bwd 3.0→2.7ms，K128 / fp32 持平（1.0ms / 2.8ms 稳定复测）。

---

## 结论（截至 Phase 3）

- 正确性：22/22 门禁全绿（fwd/bwd/varlen/3 dtype/非整块/initial_state）；
  官方 `tests/ops/test_gdn2.py` 在 NPU 下因 CUDA-required 装饰器 skip（2 passed
  26 skipped，环境限制而非回归）。
- bwd intra：统一循环 dq_db（>600us）→ 拆分配对后 ~150us/步
  （dq_past 27.2 + dq_db_diag 56.1 + dkt_future 23.6 + dk_dg 57.7）。
- fwd intra：token_parallel 102.9us → sub_chunk 78.7us。
- 已确立的 Ascend 经验：动态循环与直线 dot 混合触发 BiSheng 病态 codegen；
  拆 kernel 配对（past/future 动态 + diag 直线）+ mid-row 参考因子分解 +
  独立 BK 是该类 chunked bwd kernel 的可复用模式。
- 剩余瓶颈与停止判定：GDN2 专属 kernel 中最大为 `inter_solve_fused`
  97.6us/步（scalar 0.497，端到端占比 <4%），重写为更低 scalar 结构的预期
  收益 <50us 且风险高；其余热点（gated_delta_rule_h ~81us、gla_fwd_o ~49us、
  l2norm/cumsum 等）属于 KDA/GLA 共享管线，超出 GDN2 范围。判定为当前合理
  收敛点。

---

## Phase 4 — inter_solve_fused 深挖（NC=4→2 + 两次失败的 diag 优化）

**瓶颈画像**（Phase 3 收尾时）：`inter_solve_fused` 97.6us/步，
aic_scalar_ratio 0.497 —— NC=4（BC=16）导致 12 个 `[16,·]×[·,16]` 小 dot
+ 9 步求解链 dot，标量开销主导。

### 4a. 统一首行参考（失败，已回滚）

把 K-loop 的 per-target 参考行统一为 chunk 首行参考，exp2 12→8 次、trans
6→4 次。**数值失败**：列因子在 j>r 侧产生正指数 → `exp2(+∞)` 溢出 NaN，
bwd 全线崩溃。数学上 per-target 参考（每个目标块用自身首行）是必要的数值
稳定结构，不能为省 exp2 统一。回滚后 22/22 恢复。

### 4b. BC 16→32（NC=4→2，保留）

- **动机**：NC 减半 → inter_solve 的 12 dot + 9 求解链 dot 变 4 + 1；
  diag_solve 标量列循环 O(BC²) 仅翻倍。
- **实现**：`BC = min(_FWD_BC, BT//2)`（`_FWD_BC=32`；BT=32 时保持 16
  维持 NC=2）；`_get_sub_chunk_bk/_get_inter_bk` 接收实际 BC 做 UB 预算
  （K=64 下 BK=64 不变，预算内）。kernel 代码本身 BC 参数化无需改。
- **单 kernel 对比**（单步）：`inter_solve_fused` 97.6→**58.9us（-40%）**；
  `diag_solve` 23.8→61.1us（+157%，标量迭代 14→30）；
  `sub_chunk` 78.7→86.2us（+10%，疑似共享设备噪声，dk_dg 等未改 kernel
  同期波动 2×）。fwd kernel 合计 200.1→206.2us。
- **同进程交替 A/B**（/tmp/ab_bc.py，BC=32/16 交替 3 轮 × 30 次取
  min/med，消除跨进程设备争用漂移）：BC=32 min 0.93-0.95ms / med
  1.07-1.11ms；BC=16 min 1.04-1.07ms / med 1.23-1.34ms。
  **fwd 端到端 -11%（min）/ -14%（med）**。早前跨进程 bench 的「持平」
  读数是争用噪声误判——教训：共享 NPU 上跨 run Duration 不可比，结论必须
  用同进程交替 A/B 确认。
- **门禁**：22/22 PASS。

### 4c. diag_solve 分块求逆（失败 ×2，已回滚，教训固化）

- **尝试**：`M=[[A,0],[C,D]] → M⁻¹=[[A⁻¹,0],[-D⁻¹CA⁻¹,D⁻¹]]`，标量循环
  只求逆两个 [16,16] 半块（30→28 迭代且 tile 减半），跨项走 Cube dot。
- **变体 1（动态 trip count）**：61.1→**110.5us（+81%）**，core 从
  AI_VECTOR 变 MIX_AIC。门禁全绿但性能崩坏。
- **变体 2（固定 range(2,BH2) + load mask 处理越界行）**：117.2us，同样
  病态。
- **结论（固化经验）**：diag_solve 内**循环与 tl.dot 的任何混合**——无论
  bound 动态与否、dot 在循环内还是循环后——都触发 BiSheng 病态 codegen。
  该 kernel 必须保持纯标量。已在 kernel 内注释记录。若未来 BiSheng 修复
  该模式，重新启用分块求逆预计 diag_solve 回 ~28us（BC=32 全链 fwd
  kernel 合计可再 -16%）。

### 4d. num_warps 清理

移除 `_launch_sub_chunk_kernel`/`_launch_inter_kernel` 中的 `num_warps=2`
（及 `_NUM_WARPS_SUB/_NUM_WARPS_INTER` 常量）：Ascend Triton 不支持
num_warps/num_stages，属于移植期遗留的 CUDA 习惯，违反
fla-ascend-performance skill 硬性要求。移除后门禁/bench 无变化（该参数
在 Ascend 路径本就被忽略）。

### Phase 4 小结

| 项 | 前 | 后 | 备注 |
|----|----|----|------|
| inter_solve_fused | 97.6us | 58.9us | NC=4→2，dot/求解链 -75% |
| diag_solve | 23.8us | 61.1us | O(BC²) 标量循环翻倍，分块求逆不可用 |
| sub_chunk | 78.7us | 86.2us | 争用读数，跨 run 不可比 |
| fwd 端到端 (K64 bf16) | 1.06ms | 0.95ms | 同进程交替 A/B，min 口径 **-11%** |
| 门禁 | 22/22 | 22/22 | 每次变更后全量复测 |

判定：BC=32 保留并确认端到端收益。Cast（bwd 返回 6× `.to(dtype)`，
fp32 梯度 buffer 降 dtype）实测 13.7us/次 ≈ bwd 的 3%，profile 中 217us/次
为争用夸大 16×，非瓶颈，不动。新的最大 GDN2 kernel 为 fwd 侧
`sub_chunk`(~86us) / `diag_solve`(61us) / `inter_solve`(59us) 与 bwd 侧
`dk_dg` / `dq_db_diag`（争用读数 ~115us）。

## Phase 5 — sub_chunk 消融定位（结论：BiSheng codegen 结构性极限）

**瓶颈画像**（Phase 4 收尾时）：`sub_chunk` ~80us/步，aic_scalar_ratio
0.41~0.49，MTE/Cube 均 ~50%。为区分「dot 本身」与「dot 周边代码」的开销，
做了一轮 kernel 级消融（/tmp/ab_subchunk*.py，同进程 wall-clock min/med），
每轮只改一个变量：

| # | 变量 | 基线 | 变体 | 结论 |
|---|------|------|------|------|
| A1 | `tl.trans` 位置（先乘后转 vs 先转后乘） | 46.5 | 46.8 | 无差异，trans 非瓶颈 |
| A2 | exp2×3 → exp2×1（两因子相乘替代） | 46.7 | 46.9 | exp2 非瓶颈（AIV exp2 吞吐充足） |
| A3 | 去掉 2 个 `tl.dot`（数学不等价，仅测开销） | 135 | 63.5 | **dot 链占 kernel ~53%**（71/135 wall），为固有 FLOPs |
| A4 | 去掉 3 个 `tl.where`（含 mask 落地） | 43.3 | 36.8 | where 链 ~6.5us（15%） |
| A5 | gate 因子 fp32→bf16（乘 k 后降精度） | 43.3 | 33.2 | ~10us（23%），但改变中间精度，见下 |
| A6 | 去 b-gate（`(b*k)` 路径） | 43.3 | 38.6 | b-gate 乘法路径 ~5us |
| C1 | dot 尺寸 [32,64]×2 → [64,64]×1（pad q 拼接） | 43.3 | 46.1 | **负优化**，大 dot 无收益 |
| D2 | 2 dot → 1 dot（qk/akk 列拼接共享 kgt） | 46.1 | 43.8 | ~5%，但 join/reshape 路径 BiSheng 代码生成失败（K>32 时 collapse 限制），无法落到正式实现 |
| P1 | 持久化 core-grid task loop（24 core 自取任务） | 128.6 | 154.4 | **负优化 +20%**，与 Phase 2/4 的 dynamic-loop+dot 病态一致 |

（A3-A6/C1/D2 为缩配置单 kernel 消融，绝对值与整 kernel 80us 不同口径；
看相对差。）P1 为全量 task（NT×NC×BH=256）真实 kernel 对比，数值 diff=0。

### 判定

1. `sub_chunk` 的 ~80us 中约一半是 2 个 `tl.dot` 的固有 Cube 开销（A3），
   剩余开销集中在 fp32 gate 因子生成链（exp2/where/乘法，A4-A6 合计
   理论 ~15-20us）。
2. 所有可行的结构改写（C1 大 dot、D2 合并 dot、P1 持久化）均为 0 或
   负收益——BiSheng codegen 对该 kernel 形态已无结构性红利。
3. 唯一正收益且可落地的变体是 **A5（gate 因子 bf16 化，-23%）**，但它
   改变 validated kernel 的中间计算精度（fp32→bf16 gate），按仓库规范
   属于需要 RFC 的设计决策，且 23% 单 kernel 收益折算端到端 <2%。
   **记录不实施**，留待 RFC 讨论。
4. 该 kernel 判定为已达 BiSheng codegen 结构性极限，关闭 Phase 5。




## Phase 6 — bwd diag 融合 + FULL_TILE 消融（含测量噪声校准）

**瓶颈画像**（Phase 5 收尾时，bwd 占端到端 ~65%）：bwd intra 三 kernel
`dq_db_diag` 106.2us / `dk_dg` 107.7us / `dq_past`+`dkt_future` ~74us，
前两者 aic_scalar_ratio ~0.5、共享全部输入 load（q/k/g/b/dAqk/dAkk）、
同一 task grid、同一 mid-row 参考行。

### 6a. diag_fused 融合 kernel（保留）

- **实现**：`chunk_gdn2_bwd_kernel_intra_diag_fused_npu` 合并旧
  `dq_db_diag` 与 `dk_dg` 的对角部分——dq2/dk2/dg2/db2 的 diag 项 + dkt
  对角贡献（读 `dkt_part`）+ 输入梯度累加一次 load 完成，消除 dk2/dg2
  的 GM 往返；launch 顺序变 dkt_future → diag_fused → dq_past。
- **数学等价**：`dg2` 的 k-项 `(diag - dkt)*k` 拆入 fused、past-项
  `(past*b)*k` 拆入 dq_past，保持累加顺序重排后逐项等价；`(b_dk2 -
  b_dkt)` 必须用 diag-only dk2（乘 b 前），注意语句顺序。
- **结果**：两 kernel 213.9us → fused 单 kernel 202.7/207.3us（两次
  profile），-3~5%；门禁 22/22 PASS。收益小于预期——dk2/dg2 往返本就
  只占两 kernel 的 load 带宽小头，融合省下的是重复的 q/k/g/b load 与
  exp2 链，大头 scalar 开销未变。

### 6b. FULL_TILE 边界检查消除（消融后判定：无效，中性保留）

- **假设**：整块路径（非 varlen、T%BT==0、K%BK==0）下
  `boundary_check=(0,1)` 的逐元素谓词编成 AIV 标量地址检查，是
  scalar-bound 的来源之一。
- **微基准先行**（/tmp/micro_elem.py，K1 bc+where / K2 no-bc+where /
  K3 no-bc-no-where）：41.1/42.0/41.3us —— **boundary_check 与 where
  均非瓶颈**，exp2 elementwise 链本身才是（该形态下不可再优化）。
- **kernel 实测**：fused kernel FULL_TILE=True 202.7us vs False
  207.3us（两次独立 profile，噪声内）；`dq_past`/`dkt_future`（动态
  循环 kernel）加 FULL_TILE 无稳定差异。结论：**boundary_check 消除
  对该类 kernel 无实际收益**（谓词开销被 MTE/Scalar 流水掩盖）。
  实现保留（fused 传 FULL_TILE=整除性，动态循环 kernel 固定 False），
  标注中性。

### 6c. 测量噪声校准（重要教训）

共享 NPU 上跨进程 wall-clock 波动可达 ±9%（同配置 e2e 2.34-2.56ms），
单 kernel Duration 也有 ±5-15%（dq_past 同配置两次 profile 45.0 vs
53.9us）。期间一度把 FULL_TILE 读成 -10% 端到端收益，profile 复测后
修正为中性。**固化流程：任何性能结论必须 (1) 同进程交替 A/B 或 (2)
两次独立 profile 一致才可采信；单次读数一律视为候选假设。**

### Phase 6 小结

| 项 | 前 | 后 | 备注 |
|----|----|----|------|
| bwd diag 两 kernel | 213.9us | ~205us（fused 单 kernel） | -3~5%，去 GM 往返 |
| 端到端 fwd+bwd | 2.69ms | 2.34-2.56ms（噪声带） | 门禁 22/22 |
| FULL_TILE | — | 中性 | 微基准+profile 双重证伪假设 |

剩余瓶颈（按 profile Ratio%）：fused diag ~205us（scalar-bound，
aiv_vec_ratio 仅 0.07，fp32 elementwise 链不可再降）、
`gated_delta_rule_h` 158us/2 步（KDA 共享管线，8/24 核结构性欠载，
V-split 被 BiSheng memref 限制阻塞）、DSA/Copy/Transpose 等
框架开销。GDN2 专属 kernel 中已无 >5% 且结构可动的目标，判定为
当前合理收敛点。

## Phase 7 — 后续优化路线图（待执行）

Phase 6 收敛判定后，仍有两个已确认、未执行的优化点（本轮先归档，
后续按序推进；每项落地后在此追加实测数据）。方法论语境见
`OPT_PLAYBOOK.md`。

### 7a. db2 布局直写（优先级最高，预期 -70~80us/步 端到端）

**现状**：`chunk_gdn2_bwd_intra_npu` 结尾的收集阶段
`db2.permute(1,2,3,0,4).contiguous().reshape(B,T,H,NK*BK)[..., :K]`
+ `db.add_(...)`，bwd profile 实读框架开销：Transpose 54-64us +
Copy ~8us + VectorAdd ~6us，合计 ~70-80us/步，约占 bwd intra
链路的 15%——比 Phase 6 的 diag 融合收益（~9us）大一个量级。

**根因**：partial db2 按 `[NK, B, T, H, BK]` 排布（i_k 是最外维），
kernel 内 `db_ptr = db + ((i_k * all + bos) * H + i_h) * BK`、
块指针 shape `(T_cur, BK)`；收集阶段必须整块 permute+contiguous
才能拼回 `[B, T, H, K]`。

**方案**：让两个写 db 的 kernel（`diag_fused` 纯 store、`dq_past`
load-modify-store）直写 `[B, T, H, K]` 布局——指针改为与
dkt_part/dq 同构：

```python
db_ptr = db + (bos * H + i_h).to(tl.int64) * K      # [B,T,H,K] 基址
p_db = tl.make_block_ptr(db_ptr, (T_cur, K), (H * K, 1),
                         (i_ti, i_k * BK), (BC, BK), (1, 0))
```

NK 个 K-tile 各写各自 BK 列、互不重叠；`dq_past` 的 RMW 累加语义
不变（diag_fused 先写、dq_past 后读改写，同 stream 保序）。wrapper
删掉 permute/contiguous/reshape/slice，`db2 = q.new_empty(B,T,H,K,
float32)` 后直接 `db = db.add_(db2)`。

**数值**：不变——仍是 fp32 partial + 一次 add_，无精度影响。

**风险**：低。指针结构与 kernel 内现有 dkt_part 块指针逐字段一致，
改动约 6 行（两个 kernel 各 2 行 + wrapper 3 行）。

**验证**：门禁 22/22 + 同进程交替 A/B（e2e 与 diag_fused/dq_past
kernel Duration）+ 两次独立 profile，确认 Transpose/Copy/VectorAdd
三 op 从 top 榜消失。

### 7b. dkt_part zeros → empty（预期 -6us，可选项）

**现状**：`dkt_part = torch.zeros_like(dk, dtype=torch.float)` ~6us。
zeros 的原因：最后 sub-chunk（i_i = NC-1）没有 future 块，
`dkt_future` 不为它启动任务，这些行依赖初始化的 0。

**候选方案**（均有附加成本，执行前需 A/B）：

- 方案 A：dkt_future 覆盖全部 NC 个 sub-chunk，i_i=NC-1 的空任务
  只写 0。任务数 +`NK_dk*NT*BH`（NC=2 时翻倍），空任务仍要
  load+store，大概率负收益。
- 方案 B：仅 NC==1 形态（T <= BT 的短序列/decode）用
  `torch.empty` + diag_fused 里 `if NC > 1:` 编译期跳过 dkt_part
  load（NC 是 constexpr，分支编译期消除，dkt 项恒 0）。NC>1 形态
  保留 zeros，正确性不受影响。

**判定**：收益 6us、方案不干净，列队尾；7a 落地后重估。

### 7c. 已证伪 / 搁置项（勿重复尝试）

| 项 | 结论 | 证据 |
|----|------|------|
| exp2 快速化（libdevice / fast-math） | 不可行且无收益 | 微基准 K5：无 exp2 纯乘链 min 39.4us vs 带 exp2 41.7us，exp2 仅占 ~6%；`triton.language.extra.libdevice.exp2` 在 Triton-Ascend 3.2.0 直接编译失败（NoneType attribute error） |
| FULL_TILE 边界检查消除 | 中性 | 6b：微基准三变体 41.1/42.0/41.3us + kernel 两次独立 profile 202.7 vs 207.3us（噪声带内）；谓词开销被 MTE/Scalar 流水掩盖 |
| persistent core-grid task loop | +20% 回退 | BiSheng 动态循环+tl.dot 病态 codegen（Phase 2a，128.6→154.4us） |
| h_blockdim64 V-split（BV=32） | 编译阻塞 | BiSheng memref address-space mismatch，[64,32] store 形态不受支持 |
| KDA 共享管线（gated_delta_rule_h 158us/2步、inter_solve_fused ~59us） | 超出 GDN2 范围 | 共享代码，改动影响所有 KDA 用户，需独立评估 / RFC，不在本算子优化范围 |

### 7d. 长尾

7a 落地后重新 profile：看 Transpose/Copy/DSA 框架项剩余多少，
再重看 fused diag（~205us，scalar-bound，aiv_vec_ratio 0.07）。
若仍无 >5% 且结构可动的 GDN2 专属目标，则终态判定：
剩余时间为 KDA 共享管线 + BiSheng fp32 elementwise 结构性极限。
