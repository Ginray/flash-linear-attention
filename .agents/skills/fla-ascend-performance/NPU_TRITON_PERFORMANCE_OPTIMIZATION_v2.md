# NPU 上 Triton 性能优化方式与开源仓库案例

> 状态：v2 深挖版（2026-08-20）。本文聚焦 Ascend NPU / Triton-Ascend；在原稿的 kernel/tiling/UB/Cube-Vector 优化基础上，补充 Host↔Device 同步、metadata/JIT、选择性 specialization、live-range、离散访问规则化、950 SIMT 等近期真实仓库案例。所有版本敏感结论仍应结合目标硬件、CANN、Triton-Ascend 与 PyTorch 版本复测。

## 0. 阅读约定

本文不把“GPU Triton 上通常有效”自动等同于“Ascend NPU 上有效”。每个重要结论尽量附一手来源，并按以下证据等级标注：

- **A — 代码/官方文档证据**：可定位到官方文档、仓库源码、测试或提交。
- **B — 仓库案例证据**：可定位到真实开源实现，但缺少独立性能数据或只覆盖有限形状。
- **C — 待实测假设**：符合硬件/编译原理，但必须用目标环境的 benchmark 和 profiler 验证。

补充一个重要区分：**kernel 机制被官方/源码验证，不等于 wrapper/serving 路径也已经端到端最优**。例如官方示例里的 kernel 可能是最佳实践，但 Python wrapper 仍可能包含 `.item()` / `.cpu()` / metadata 重建等同步或调度开销。本文在引用仓库案例时尽量区分“kernel 机制证据”和“端到端路径证据”。

性能优化的硬门槛：

1. 正确性不回退；归约、混合精度和边界形状要单独验证。
2. 不用只对单一 shape 有效的硬编码破坏原有泛化性。
3. 同时报告端到端耗时和设备 kernel 耗时，区分编译、host 调度、布局转换与 kernel 本体。
4. 小、中、大规模以及对齐/非对齐边界分别测量；只比较同一设备、同一软件栈和同一同步口径下的数据。

## 1. 优化总图

Ascend NPU 上的 Triton 优化不是单独调大 `BLOCK_SIZE`，而是同时处理“核数、核内 tile、片上空间、数据布局、执行流水、算法和编译选项”。推荐按下列顺序推进：

| 顺序 | 先回答的问题 | 常用动作 | 失败信号 |
|---|---|---|---|
| 1. 建基线 | 慢在冷编译、Host、同步、kernel，还是布局转换？ | warmup、同步、端到端与 kernel 分开计时 | 首次编译混入耗时；异步计时虚低 |
| 2. 查 Host/runtime | 是否存在隐式 D2H scalar sync、metadata 重建、graph break、JIT 变体爆炸？ | 审查 `.item/.cpu/.any/.max`、预构建 metadata、统计 JIT variant/cache | kernel 很短但 Host gap 大；每层/每步重复同步或构造 tensor |
| 3. 定设备瓶颈 | 最长的是 MTE、Scalar、Vector 还是 Cube 流水？ | `msprof op`、`PipeUtilization.csv`、仿真流水图 | 只看总耗时，盲调 tile |
| 4. 改算法/数据流 | 是否存在多 pass、冗余 GM 往返、可融合或应拆分的阶段？ | 单 pass、在线算法、融合、按 live-range 拆分、UB 内复用 | kernel 数多；或单 kernel 峰值活跃集过大 |
| 5. 改调度 | 独立任务是否填满物理核？logical grid 是否又远超物理核？ | 在“过少并行”与“过细多 wave”之间调 grid；必要时核内 stride loop | grid=1/少量 program 串行大量 batch；或 Block Dim 远大于核数 |
| 6. 改访存 | 是否连续、对齐、搬运粒度足够大？离散访问能否先规则化？ | 连续 offset、合并小搬运、bin/group/pad、布局重排 | MTE 指令多、带宽低、流水频繁断流 |
| 7. 控片上空间 | 活跃 tensor、mask、index、临时量和多缓冲是否溢出？ | 二级 tile、缩短 live-range、constexpr DCE、按需关闭/限制 multibuffer | UB/L1 overflow、tile 增大反而变慢 |
| 8. 用硬件单元 | `tl.dot` 是否真正走 Cube？Vector 是否退化为 Scalar？ | 调整 M/N/K tile、dtype、比较/索引表达式 | Cube 利用率低、Scalar/FLOWCTRL 饱和 |
| 9. 控 specialization/compiler | 哪些参数应动态复用，哪些结构模式应编译期专化？ | `do_not_specialize`、`tl.constexpr`、shape bucket、compiler options | 重编译多；或运行时分支让多条数据流同时占 UB |
| 10. 联合搜索 | shape、dtype、SoC 下最佳参数是否不同？ | `triton.autotune` / `max_autotune` 搜 tiling 与 NPU 编译选项 | 单一配置只在少数 shape 上快 |

这条路径与当前主线官方文档的分层一致：先做多核任务并行，再做单核数据搬运与单核计算，并把 `BLOCK_SIZE`、子块及 NPU 编译参数纳入 autotune。[证据 A：官方编程指南](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/zh/programming_guide/index.md)

从工程视角可以把优化对象进一步拆成五层，避免只盯 kernel：

```text
1. System / Host Wrapper
   D2H sync、graph、metadata、layout preparation
            ↓
2. Launch / Specialization
   grid、JIT variant、constexpr、do_not_specialize、cache
            ↓
3. Algorithm / Data Layout
   fusion/split、online、bin/group/pad、contiguous representation
            ↓
4. Kernel
   tiling、UB、MTE、Vector/Cube、reuse、pipeline
            ↓
5. Compiler / Architecture
   multibuffer、CV balance、AutoBlockify、950 SIMT/专用扩展
```

近期 SGL Kernel NPU、vLLM-Ascend、Flash-Linear-Attention 的高收益改动已经明显覆盖第 1–2 层，因此“只 profile 设备 kernel”不足以指导服务框架中的完整优化。

## 2. 先测量：从现象定位瓶颈

### 2.1 计时口径

1. **首次编译与稳态执行分开**：先 warmup，确认缓存命中后再重复 launch；同时保留冷启动数据，但不要拿它代表 kernel 性能。
2. **异步执行必须同步**：计时区间结束前调用 NPU synchronize；官方 launch 示例也在读取结果前显式同步。
3. **至少报告两层时间**：
   - 设备 kernel：回答 kernel 本体有没有变快。
   - 端到端：包含 Python/dispatcher、布局转换、临时 tensor、同步和多 kernel launch，回答实际调用有没有变快。
4. **统计分布而非单点**：warmup 后多轮测量，至少保留 median 与高分位；固定频率策略、设备占用和软件栈。
5. **按 shape bucket 测**：小/中/大、2 的幂/非 2 的幂、对齐/尾块、连续/跨步、不同 dtype 分开统计。
6. **同步点单独统计**：除了最终 synchronize，还要检查调用路径中是否发生 device scalar 回读；这类同步可能把设备流水切断，但不一定出现在目标 kernel 的设备时间里。
7. **记录 JIT/cache 行为**：高频动态 shape/kernel 除 steady-state 外，记录编译 variant 数、cache hit、首次/新增 shape 的编译峰值。

### 2.2 用 msProf 从流水反推源码

官方推荐用 `msprof op --kernel-name=<kernel>` 采集指定 Triton kernel，并从 `op_summary_*.csv` / `PipeUtilization.csv`、仿真 `trace.json` 或 `visualize_data.bin` 分析。需要把仿真指令关联到 Triton 源码时，设置 `TRITON_DISABLE_LINE_INFO=0` 后重新编译。[证据 A：官方 profiling 指南](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/zh/debug_guide/profiling.md)

| 观测 | 优先怀疑 | 下一步检查 |
|---|---|---|
| `aiv_scalar_ratio/time` 或 FLOWCTRL 高 | 循环内整除/取模/比较、标量地址计算；不支持的 dtype 使向量运算退化 | 仿真流水 + 源码热点；检查 i64/i32 compare、复杂索引 |
| MTE2/MTE3 长、有效带宽低 | tile 太小、非连续/未对齐、相同数据重复搬运 | 计算理论字节数；比较实际指令数与理想大块 DMA 数量 |
| Vector 利用率低且有周期性空洞 | tile 太小、搬运与计算串行、依赖/同步阻断 | 增大有效工作块；检查 multibuffer 和数据依赖 |
| Block Dim 远大于物理核数 | logical grid 过细，Host 启动/初始化轮次过多 | 压缩 grid，并在核内 stride-loop 多个 tile |
| Block Dim 很小但 program 内串行大量独立 batch/row | 并行度不足，大量 Vector/Cube Core 空闲 | 把独立任务提升到 grid 维度，直到能填满有效物理核 |
| Host gap 明显、设备流水被切断 | `.item()` / `.cpu()` / `.tolist()` / device `any/max/min` 等 D2H scalar sync | 使用 scheduler 已有 CPU metadata，或预构建并复用 Host-side metadata |
| Cube 低、Vector/Scalar 高 | `tl.dot` 的形状、布局或 dtype 未匹配硬件路径；CV 后处理成为长板 | 查看 Cube 流水和编译 IR；分别调 Cube tile 与 Vector 后处理 |

先算理论下界再解释 profiler：搬运下界约为“总字节数 / 对应带宽”，计算下界约为“操作量 / 对应峰值算力”。MTE2 与 MTE3 同时访问 GM 时共享带宽，小块搬运通常达不到峰值；因此不能把各流水的独立理论下界简单相加或假设都能满带宽。[证据 A：官方 profiling 指南](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/zh/debug_guide/profiling.md#理论参数)

### 2.3 一个已确认的 Scalar 退化案例

主线 profiling 文档给出的 LayerNorm 例子中，`tl.where(cols < N, ...)` 的 `cols` 为 i64；该比较在示例硬件路径上无法向量化而退化为 Scalar。把用于 `tl.where` 比较的值转为 FP32 后，可生成 Vector cast/compare。这个改写**不是普遍要求**：文档同时指出 `tl.load` / `tl.store` 的 mask 大多能由编译器自动向量化，应根据热点和目标版本决定。[证据 A：官方 i64/i32 compare 案例](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/zh/debug_guide/profiling.md#示例i64i32-的compare在npu上无法启用vector导致向量计算转为标量计算)


### 2.4 Host↔Device scalar sync：优先排查“看起来只拿一个数”的操作

在 serving / varlen / scheduler 场景中，下面这些 Python 写法如果作用于 NPU tensor，往往意味着 Host 必须等待设备产生结果：

```python
value = tensor.item()
value = tensor[-1].cpu().item()
max_len = seqlens.max().item()
flag = bool(has_initial_state.any())
ids = tensor.tolist()
```

它们的代价不是“拷贝 4/8 字节”，而是可能形成：

```text
previous NPU work → D2H scalar dependency → CPU wait → next launch
```

SGL Kernel NPU 的 Qwen3.5/GDN prefill 优化 PR #454 直接把 `query_start_loc[-1]`、`seqlens.max()`、`has_initial_state.any()` 以及 FLA chunk metadata 的 device scalar 回读替换为调用方已经持有的 `seq_lens_cpu` / `cu_seqlens_cpu` / `max_query_len` / `has_initial_state_any`，并支持传入预构建的 `chunk_indices/chunk_offsets`，目的就是避免每层 stream stall。[证据 B：SGL Kernel NPU PR #454](https://github.com/sgl-project/sgl-kernel-npu/pull/454)

通用检查原则：

- scheduler/dispatcher 本来已经在 CPU 上知道的长度、batch、是否有状态，不要再从 NPU tensor 反向查询；
- `cu_seqlens`、chunk mapping、block mapping 在多层间不变时，优先构建一次并复用；
- graph capture 路径中 device→Host 的动态控制流通常也是 graph break 风险；
- 保留 legacy fallback 可以兼容旧调用方，但 benchmark 应明确测“纯 Host metadata 快路径”和“device scalar fallback”两种路径。

这类优化必须算端到端，单看目标 kernel 的设备时间可能完全看不到收益。

## 3. 数据搬运与内存访问优化

### 3.1 连续、对齐、足够大的事务

- 让同一程序处理连续地址：`base + tl.arange(0, BLOCK)` 通常比带大 stride 的逐元素 gather 更容易形成大块搬运。
- 尾块保留 mask，不能为了对齐而越界。
- 当前迁移指南要求 Vector 算子检查 **32B** 访存对齐，Cube-Vector 融合算子检查 **512B** 对齐；这是目标后端约束，不应把 GPU 上的对齐经验原样照搬。[证据 A：GPU→NPU 迁移指南](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/en/migration_guide/migrate_from_gpu.md#check-single-program-data-transfer)
- 同样的总字节数，很多小事务通常显著慢于少量大事务。若 profiler 显示 MTE 指令数远高于 `总字节数 / 理想事务大小`，优先重做布局/tiling，而不是继续堆计算优化。

### 3.2 离散访问规则化：不要只优化单条 gather

离散访问可以按三个层级处理：

1. **直接 GM random access**：实现最直接，但大量小事务最容易让 MTE 效率下降。
2. **连续候选区 → UB gather/select**：对候选工作集较小、索引复用较多的 gather/select，先连续加载候选区到 UB，再用 `tl.gather` 在片上选择。官方 `pick_kernel` 示例就是这一方式。[证据 A：官方 UB-select 示例](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/zh/programming_guide/index.md#先将数据搬运到ub上再从ub中select目标值)
3. **先重组织 index，再做批量连续搬运**：MoE/MegaBlocks 一类场景可按 expert/bin 对 token 先 group/bin/pad，使原本完全离散的访问变成“低维分组 + 高维连续”的块操作。`triton-ascend-ops` 的 `gather_scatter`、`binned_gather_scatter`、`padded_gather_scatter` best-practice 就体现了这种思路，并实际使用 `tl.multiple_of` / `tl.max_contiguous` 等编译提示表达对齐/连续性约束。[证据 A：triton-ascend-ops best_practice](https://github.com/Ascend/triton-ascend-ops/tree/755cf18c30f18720f67b6360c2b2856b64739822/tutorial/best_practice)

适用边界：候选工作集必须放得进片上空间，或分组/重排成本必须小于随机访问节省。索引极稀疏、候选区很大、一次性使用时不一定成立，应把“额外预取字节、重排成本、MTE 指令数、UB 占用”一起测。对于 Ascend 950，还要把 SIMT 间接访存/专用 gather 路径纳入候选，不能把 A2/A3 的 UB-select 经验写成跨代硬规则。

### 3.3 减少 GM 往返

- 多个逐元素/归约步骤若生产者—消费者关系紧密，优先在一个 kernel 中融合，让中间值留在片上。
- 归约/归一化若一整个有效工作块能驻留 UB，优先一次 load 后完成统计与输出；双 pass 或三次读取同一输入只在片上空间不足、数值算法或依赖确有需要时采用。
- 为了连续访问而在 Host 侧做 `contiguous()` 可能有利于 kernel，却可能拖慢端到端。布局转换成本必须计入，并考虑缓存/复用转换后的布局。

以上三项的收益属于 shape 和版本相关假设；没有真实 profile 时按 **C** 级处理。

### 3.4 复用 tile：让多个 row/sequence 共用一次 invariant load

除了“核间 tile + 核内 sub-tile”，还可以单独考虑 **reuse tile / batch tile**：一个 program 内处理多个逻辑实例，但共享一次 weight、bias、lookup table、cos/sin、静态 descriptor 等加载。

```text
load invariant once
      │
  ┌───┼───┐
 row0 row1 row2 ...
```

这类优化适合 LayerNorm/Conv/RoPE 等“参数小、每 row 重复读取”的算子。收益来自减少 GM/MTE 事务和重复标量地址计算，但 batch tile 过大也会增加 UB live set、延长单 program 时间并降低负载均衡，仍需结合物理核数与 UB 搜索。Flash-Linear-Attention 的 Ascend fused norm/gate 优化通过 BT-tiled program 摊薄 weight/bias load，就是该模式的一个实际案例。[证据 B：FLA PR #1044](https://github.com/fla-org/flash-linear-attention/pull/1044)

## 4. Tiling、任务调度与核间并行

### 4.1 grid 并行度要“双向调节”，不是机械压到物理核数

当前主线推荐优先用 1D grid；纯 Vector 算子按 Vector Core 并发能力组织，包含 `tl.dot` / CV 融合的算子按 AI/Cube Core 并发能力组织。但调度目标不是“永远 grid=物理核数”，而是：

> **先让足够多的独立任务填满有用硬件，再避免大量极细 logical program 带来的多 wave、初始化和调度开销。**

存在两个相反的失败区：

```text
grid 太小                         grid 远大于物理核且每 tile 很小
大量独立工作塞在 program 内串行     大量 wave / program 初始化 / 调度
        ↓                                  ↓
增大 grid，把独立任务暴露出来         压缩 grid + 核内 stride-loop
```

当 logical tile 多于物理核且每 tile 很小时，可用每核跨步循环：

```python
pid = tl.program_id(0)
num_programs = tl.num_programs(0)
for tile_id in range(pid, num_tiles, num_programs):
    ...
```

这样把并发 program 数控制在合理范围，剩余 tile 在核内分批处理。物理核数应通过设备属性查询，不要写死。[证据 A：官方多核任务并行指南](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/zh/programming_guide/index.md#通用多核任务并行)

反过来，如果代码是 `grid=[1]`，却在 kernel 内 `for req_idx in range(batch_size)` 串行处理互相独立的请求，就应该增大 grid。vLLM-Ascend PR #14066 将 spec-decode 输入 copy kernel 从单 program + batch 内循环改为 `grid=[batch_size]`、每个 program 处理一个 request，在 Ascend 910B2C 的报告 workload 上 kernel 平均时延从约 1035 µs 降到 154 µs，端到端吞吐 433→638 tok/s（+47%）。这是“并行度不足时反向增大 grid”的直接例子。[证据 B：vLLM-Ascend PR #14066](https://github.com/vllm-project/vllm-ascend/pull/14066)

`TRITON_ALL_BLOCKS_PARALLEL=1` / AutoBlockify 仍可自动处理超大独立 grid；但 tile 间有执行顺序依赖时可能死锁。能显式写清任务映射时优先手工表达，动态/超大 grid 再评估自动 blockify。[证据 A：环境变量参考](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/en/environment_variable_and_compiler_options_reference.md#environment-variable-reference-table)

### 4.2 两级 tiling：核间 tile + 核内 sub-tile

- 核间 tile 决定并行度和每核总工作量。
- 核内 sub-tile 决定单次活跃数据量、搬运粒度和能否存算重叠。
- reuse/batch tile 决定同一 program 内多少 row/sequence 共享 invariant 数据加载；它与 sub-tile 是不同维度。
- 在不溢出片上空间的前提下，增大 sub-tile 通常能提高计算/访存比并减少小搬运；过大则增加 UB/L1 压力、mask/index 临时量，甚至因 multibuffer 把活跃缓冲复制多份而溢出。

官方 masked-fill 例子通过加入 `BLOCK_SIZE_SUB`，让大 logical block 在核内分段处理长序列；编译报出的 UB overflow 会显示需求量和硬件可用量，可据此缩小单次搬运量。[证据 A：官方 Tiling/UB overflow 指南](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/zh/programming_guide/index.md#tiling优化)

### 4.3 用 autotune 搜“组合”，不是单个 BLOCK

- 基础 `triton.autotune`：按决定工作集的 shape key 搜 `BLOCK_SIZE`、`BLOCK_M/N/K`、`BLOCK_SIZE_SUB`。
- Ascend 扩展 `max_autotune`：还可联合搜索 `num_stages`、`enable_hivm_auto_cv_balance`、`tile_mix_vector_loop`、`enable_ubuf_saving` 等硬件相关选项。
- key 过粗会把不同性能区间错误共用一套配置；key 过细会增加编译和缓存变体。建议用 shape bucket，而不是把每个动态长度都作为独立 key。
- autotune 配置必须先过正确性和 UB/L1 可编译门槛；对写入/原地 kernel 要处理重复运行的副作用。

[证据 A：官方 autotune 与 max_autotune 指南](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/zh/programming_guide/index.md#triton-autotune-自动调优)


### 4.4 选择性 specialization：动态参数少编译，结构模式反而要编译期专化

`do_not_specialize` 和 `tl.constexpr` 不是互相冲突，而是解决两类不同问题：

- **运行时数值会频繁变化，但不改变数据流结构**：例如 batch/token 数、某些 stride/长度。优先考虑 `do_not_specialize`、shape bucket，减少 JIT variant explosion。vLLM-Ascend 的 RMSNorm 等实现已有类似做法。
- **会改变完整 dataflow / UB live set / DMA 路径**：例如 bulk vs tail、是否有 optional state、fast path vs general path。此时反而应考虑 `tl.constexpr`/host dispatch，让编译器把不可能路径 DCE 掉。

同样，`next_power_of_2`、tile 上限、静态模式判断等能在 Host 侧一次算出的结构信息，不要每个 program/每轮循环重复做。高频 serving kernel 应同时记录 JIT variant 数、cache hit 和新增 shape 的编译峰值，而不是只看 steady-state kernel 时间。

## 5. Cube / Vector 计算优化

### 5.1 Vector：减少 Scalar 退化和控制表达式开销

- 对不影响语义与范围的索引、长度、offset，优先评估 `int32`；官方 Vector 指南明确提示不同整数类型在 Ascend Vector 路径上的支持/性能不同。但必须先证明不会溢出，不能机械把大地址或累计长度降成 int32。[证据 A：Vector 算子指南](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/zh/programming_guide/vector_operator.md)
- profiler 若显示 Scalar/FLOWCTRL 饱和，审查内层循环的整除、取模、动态分支、复杂 mask、动态地址生成以及 i64/i32 compare。把可在 Host/编译期算出的值改为 `tl.constexpr` 或 launch 参数，把重复标量表达式提到循环外。
- `tl.where` 会同时求值两侧表达式；它适合数据选择，不等于控制流短路。若一侧非常昂贵，先用数据布局/算法消除无谓工作，再考虑 mask。
- Gather/Scatter 不要只追求少读字节：若 GM 离散事务很碎，连续批量搬入 UB 后片上选择可能更快；是否成立取决于候选工作集和索引复用。
- **把长 scalar loop/store 向量化**：padding、fill、metadata 初始化、简单 copy 如果逐元素 `for` 执行，容易形成 Scalar/FLOWCTRL 长尾；优先改成 `tl.arange + mask + vector load/store`。vLLM-Ascend PR #14499 报告将 8120 个 padding entry 的标量 store 改为块向量 store 后，目标 kernel 约 1140 µs→43 µs。[证据 B：vLLM-Ascend PR #14499](https://github.com/vllm-project/vllm-ascend/pull/14499)
- **减少 loop-carried 地址依赖**：在循环中反复 `ptr += offset` 既可能增加标量依赖，也更难让编译器识别 affine addressing；可优先重建 `local_ptr = base + task_id * stride + ...`。FLA 多个近期 Ascend PR 都显式采用 pointer rebinding。[证据 B：FLA PR #1149](https://github.com/fla-org/flash-linear-attention/pull/1149)

### 5.2 Cube：先建立干净的 `tl.dot` 核心

1. 明确 `A[M,K] × B[K,N] → C[M,N]` 的 stride、layout、输入 dtype、累加 dtype和输出 dtype。
2. 用 `BLOCK_M/N/K` 控制 A/B tile，沿 K 循环 `tl.dot`；当前官方示例以 FP32 accumulator 累加，再在写回前转换输出 dtype。
3. 先对纯 GEMM 核心 profile，确认 Cube 流水确实被使用；再加 bias、scale、activation、mask 或归约，否则难以判断是 GEMM 还是后处理拖慢。
4. 联合搜索 M/N/K tile 与 `multibuffer`。更大的 tile 可能提高 Cube 利用率，也会同步增加 A/B、accumulator、mask 及布局临时量的片上占用。
5. 不规则 K/V 或稀疏输入先重排成 `tl.dot` 友好 tile。官方复杂 Cube 指南建议：低维离散、高维连续的缓存访问可按连续维搬入，再用 `extension.insert_slice`/transpose 重组。
6. **Ascend `tl.dot` operand 生命周期要按目标版本验证**：FLA 在当前 Triton-Ascend 路径中反复遇到 `tl.dot(lhs, rhs)` 后继续复用 lhs 对应 UB tile 会产生错误结果，因此通过重新 load 或 `lhs + 0.0` 保留 pristine copy。这个现象不能当成 CUDA Triton 的通用语义，也不应跨版本武断外推，但对当前 Ascend backend 是需要加入 correctness test 的 compiler/backend hazard。[证据 B：FLA PR #1113](https://github.com/fla-org/flash-linear-attention/pull/1113)，[PR #1149](https://github.com/fla-org/flash-linear-attention/pull/1149)

[证据 A：Cube 算子指南](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/zh/programming_guide/cube_operator.md)

这里不把“所有 `BLOCK_M/N/K` 必须为 16 的倍数”写成跨版本硬规则：实际合法/高效粒度受 dtype、SoC 和当前编译器 lowering 约束，应以目标版本文档、编译结果和 profiler 为准。

### 5.3 CV 融合：减少 kernel 边界，但防止 Vector 尾巴拖住 Cube

- 适合融合：同一输出 tile 上的轻量 bias、scale、activation、cast，以及可局部完成的 mask/softmax/归约。
- 谨慎融合：需要跨 Cube tile 共享状态、巨大 FP32 accumulator、复杂离散重排或同步的 Vector 逻辑；必要时拆 kernel 或使用 workspace。
- 大 accumulator 的 Vector 后处理先普通分块；`extension.parallel(..., bind_sub_block=True)` 是更强、对硬件/编译配置敏感的路径，当前官方指南不建议默认启用。
- profile Cube/Vector/MTE 的等待关系：Cube 等 Vector 时缩小/简化 Vector 后处理，或 autotune CV balance；Vector 等搬运时先处理离散访存、尾轴 padding 与 multibuffer。
- CV grid 通常按 Cube Core 数组织，不沿用 GPU 的大 grid。

[证据 A：CV 融合算子指南](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/zh/programming_guide/cv_fusion_operator.md)

## 6. UB/L1/L2 占用与流水优化

### 6.1 片上预算按“峰值活跃集”计算

预算不能只算输入/输出数据块，至少要包含：

- 同时活跃的输入、输出、FP32 accumulator；
- 广播后 offset、index、mask、临时转置/重排 tensor；
- 归约中间量和 dtype cast 后的副本；
- multibuffer / workspace multibuffer 复制的缓冲；
- 编译器 lowering 额外产生的局部 buffer。

以 Atlas 800T/I A2 为例，官方当前文档给出的 UB 容量是 192 KB，并指出默认 double buffer 会显著压缩单份 tile 可用空间；这是**A2 示例，不是所有 NPU 的常量**。编译错误会报告所需与可用 bits，当前环境也可用 `ENABLE_PRINT_UB_BITS=1` 辅助查看用量。[证据 A：A2 tiling 说明](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/zh/programming_guide/index.md#tiling优化)，[证据 A：环境变量参考](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/en/environment_variable_and_compiler_options_reference.md)

实践上给编译器临时量留余量，并把“tile 大小 × buffer 倍数 × dtype”纳入 autotune 的合法性过滤；不要把报错中某台机器的可用 bits 抄成全局常量。

除了减少 tensor 数，还要做 **live-range optimization**：能提前 `store` 的结果尽量提前写回，让对应临时量尽早死亡；不同阶段需要的临时集合差异很大时，拆 kernel 可能比继续缩小 tile 更好。FLA 的 fused LayerNorm/RMSNorm+gate backward 就通过提前 store gate grad 来释放临时量，并按 forward/backward 不同峰值分别选 BT；报告的一个 Ascend910、D=1024 案例从约 14.9 ms 降到 6.0 ms。[证据 B：FLA PR #1044](https://github.com/fla-org/flash-linear-attention/pull/1044)

### 6.2 存算重叠和 load 发射顺序

`multibuffer` 常用于乒乓/多缓冲，让下一批搬运与当前批计算重叠；但**不能再写成跨场景无条件“默认 True”**。当前官方 compiler-option reference 明确注明：`multibuffer` 支持 True/False，且 **910_95 compilation scenarios 默认关闭**。因此实验报告应记录实际 SoC/编译场景和显式 option，而不是依赖默认值假设。无论默认值如何，它都要求循环存在可流水化的独立迭代，并增加片上占用；小工作集、强依赖或 UB 紧张时可能无收益甚至无法编译，应把开/关作为候选。[证据 A：当前编译选项表](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/environment_variable_and_compiler_options_reference.md#compiler-option-reference-table)

编译器不会总能替开发者重排源码中的 load。官方 `005-load_order` 案例里，循环先 load B，而 B 依赖上轮 store B，导致后面的独立 load A 也无法提前；把 load A 移到 load B 前后，A 的搬运可以与上一轮 store B 重叠。通用做法：画出循环携带依赖，把**无依赖的下一批 load 尽早发射**，但不要跨越会改变别名/可见性的 store。[证据 A：`005-load_order` 说明](https://github.com/Ascend/triton-ascend-ops/blob/755cf18c30f18720f67b6360c2b2856b64739822/tutorial/basic/005-load_order.zh.md)

### 6.3 L1/L2：优先优化可控的数据流

Triton 源码层面对 L1/L2 的显式控制有限，当前更可靠的抓手是：

- 让相邻工作重用相同的 A/B/K/V 区域；
- 避免多核同时冲刷同一缓存集合的调度顺序；
- 减少中间张量写回 GM；
- 用 profiler 验证 MTE 等待/带宽，而不是仅凭“理论上有缓存复用”判断。

对角线/分组 grid 调度、持久化 tile 等策略属于 **C** 级候选，只有在具体 GEMM/attention 的 L2 数据和目标 SoC 上验证后才升级。


### 6.4 用 `tl.constexpr` 拆 dataflow，让 DCE 真正降低 UB 峰值

运行时 `if` 虽然只执行一个分支，但 lowering 后两条路径对应的局部 buffer 可能同时进入 peak live set。对于 bulk/tail、aligned/masked、optional-state 等结构模式，可以在 Host 侧拆成不同 compile-time mode：

```python
@triton.jit
def kernel(..., TAIL_MODE: tl.constexpr):
    if TAIL_MODE == 0:
        # bulk/block_ptr-only path
        ...
    else:
        # masked tail path
        ...
```

FLA 的 Ascend `causal_conv1d` PR #1126 就发现运行时选择 `block_ptr` vs masked load 会让两套 DMA 路径同时占用 UB；改为 `TAIL_MODE` constexpr 并分开 launch 后，编译器可以 DCE 未使用路径，从而释放 UB、允许更大的 tile，并改善 Vector 利用率。[证据 B：FLA PR #1126](https://github.com/fla-org/flash-linear-attention/pull/1126)

这条原则与 `do_not_specialize` 并不矛盾：**不改变结构的动态值避免专化；改变完整 dataflow 的结构模式主动专化。**

### 6.5 Fusion 与 split 都要按“峰值活跃集”决策

减少 kernel launch/GM 往返不代表越融合越好。一个 monolithic kernel 若让多个阶段的临时量生命周期重叠，会把 UB 上限拉到各阶段 live set 的并集；拆开后每个 kernel 可以采用完全不同的 BC/BK/BV。

FLA KDA Ascend backward PR #1130 就按阶段/UB 峰值把原大 kernel 拆成多个独立 kernel，并分别扩大 tile；而 PR #1149 又在适合的阶段把 mask/mid/finalize 融合，使 `dA_acc` 留在 UB、删除额外 HBM round-trip。说明真正原则是：

> **以 live-range 和 GM round-trip 的总成本决定 split/fuse，而不是把“fusion”本身当目标。**

[证据 B：FLA PR #1130](https://github.com/fla-org/flash-linear-attention/pull/1130)，[PR #1149](https://github.com/fla-org/flash-linear-attention/pull/1149)

## 7. 算法、融合与数值实现优化

### 7.1 优先减少 pass 和中间张量

- **归约/归一化**：工作集能驻留片上时，一次 load 后完成 sum/max/variance 和 normalize，避免为了每个统计量重新读 GM。
- **Softmax/Attention**：长序列用分块 online max/sum 更新，不物化完整 score 矩阵；QK、softmax、PV 能在一个 CV kernel 中稳定流水时再融合。
- **逐元素链**：融合 activation、bias、scale、cast，减少 kernel launch 和中间 GM store/load。
- **稀疏/离散访问**：算法级改变数据组织（按 expert/bin/token 分组或预取连续候选区）通常比微调单条 gather 更重要。

融合的退出条件：片上峰值活跃集过大、同步复杂、编译变体爆炸，或端到端因 Host 侧布局扩展抵消 kernel 收益。反过来，拆 kernel 的退出条件是新增 launch/workspace/GM round-trip 大于释放 UB 和扩大 tile 的收益；二者都必须以端到端和 profiler 结果决定。

### 7.2 数值策略也是性能约束

- `tl.dot` 和大归约通常需要较高精度累加；当前官方 matmul 示例使用 FP32 accumulator。优化后按算子语义验证 rtol/atol，不用一个统一阈值掩盖本应逐位相等的整数/索引结果。
- Online softmax 必须维护稳定的 running max 与归一化因子；不能为少一次 exp 或 cast 破坏长序列稳定性。
- int32 索引优化前证明范围；FP32 compare 替代整数 compare 前确认转换保持比较语义（大整数不能无损表示）。
- 近似数学函数、FMA 融合、溢出/饱和模式都可能改变结果；分别测试 NaN/Inf、极值、负零和边界 shape。

### 7.3 只在“真实端到端”里判断 Host 侧预处理

`transpose/contiguous/expand` 可以把离散或短尾轴访问变成连续访问，但会增加 Host 调度和 GM 字节数。可接受的三种情况：预处理可被多个 kernel 复用、布局本就能由上游产出、或 kernel 收益显著高于一次转换成本。否则保留融合内重排或专门布局 kernel 作为候选一起测。


### 7.4 Metadata / descriptor 预构建与跨层复用

serving 场景里很多对象并不是“模型数据”，而是稳定的调度 metadata：

- `chunk_indices` / `chunk_offsets`；
- KV group 的 `data_ptr/stride/block_size` descriptor；
- block table / slot mapping 参数；
- expert/bin mapping；
- 某些固定 graph buffer 的 shape/offset 信息。

如果这些值在几十层、多个 decode step 或同一 graph bucket 内不变，就不应每次重新创建 tensor/descriptor。SGL Kernel NPU PR #454 支持 caller 传 `prebuilt_meta`；FLA PR #1132 允许 caller 直接传预计算 `chunk_indices`，issue reporter 报告其项目中可获得约 25% 收益；vLLM-Ascend PR #13340 则把 multi-group slot-mapping 的多组 pointer/stride/block 参数在初始化阶段构建一次并复用，PR 描述中指出旧路径每次构造参数 tensor 约有 5 ms/call 开销。[证据 B：SGL #454](https://github.com/sgl-project/sgl-kernel-npu/pull/454)，[FLA #1132](https://github.com/fla-org/flash-linear-attention/pull/1132)，[vLLM-Ascend #13340](https://github.com/vllm-project/vllm-ascend/pull/13340)

需要注意：缓存 metadata 会引入生命周期和一致性契约。只有真正静态或能按 graph/shape bucket 正确失效的数据才应缓存；不要为省构造开销复用已经与当前 batch 不匹配的 pointer/offset。

## 8. 编译器、Triton-Ascend 特有 API 与版本差异

### 8.1 当前主线可联合调优的选项

本文原稿以主线提交 `5cdbf25b` 为主要复现实验锚点；v2 同时复核了 2026-08-20 可见的主线 compiler-option reference。由于该表会继续演进，复现实验仍应记录 commit permalink。当前性能相关选项包括：

| 选项 | 作用 | 何时尝试 | 主要风险 |
|---|---|---|---|
| `multibuffer` | ping-pong/double-buffer 流水；默认值依编译场景，910_95 当前默认关闭 | 有规则循环且 MTE/计算可重叠 | 增加片上占用；默认值版本敏感 |
| `limit_auto_multi_buffer_only_for_local_buffer` / `limit_auto_multi_buffer_of_local_buffer` | 限制自动 multibuffer 的 buffer 范围 | UB/L0C 压力来自自动多缓冲 | 过度限制可能失去流水收益 |
| `enable_hivm_auto_cv_balance` | 自动平衡 CV | Cube/Vector 等待不均 | 仅 CV；版本敏感 |
| `tile_mix_vector_loop` / `tile_mix_cube_loop` | 切分 CV 的 Vector/Cube loop | 某一侧成为长流水 | 搜索空间增大 |
| `set_workspace_multibuffer` | workspace 使用 2/4 份多缓冲 | 有 workspace 且可流水 | 内存占用与同步复杂度 |
| `enable_auto_bind_sub_block` | 自动绑定 Vector 子块 | CV 后处理并行不足 | 目标硬件/编译器相关 |
| `sync_solver` / `unit_flag` | HIVM/ Cube-output 同步相关策略 | profiler 指向 CV 同步或等待问题 | 改变同步行为，必须做完整正确性验证 |
| `inject_barrier_all` / `inject_block_all` / `disable_auto_inject_block_sync` | 控制 barrier/block sync 注入 | 明确定位到同步 pass 行为时 | 错误设置可能导致性能回退或依赖问题 |
| `enable_nd2nz_on_vector` | Vector 路径 ND→NZ 布局变换 | CV layout 开销显著 | 转换本身可能更贵 |
| `enable_linearize` | 控制 linearization pass | 复杂 layout/index lowering | 默认值随版本变化 |
| `stream` | 指定 NPU stream | 多流/框架集成需要显式控制 | 时序与同步契约更复杂 |
| `auto_blockify_size` / `TRITON_ALL_BLOCKS_PARALLEL` | logical grid 自动 blockify | 超大且互相独立的 grid | 有顺序依赖可能死锁 |
| `compile_mode`（Ascend 950） | SIMD、混合、SIMT-only | 离散/非结构化访存 | 仅 950；不能外推到 A2/A3 |

不要一次全开；把 profiler 指向的 1–3 个选项与 tile 联合搜索，并保存“软件版本 + SoC + shape key + 最优配置”。官方当前选项表还包含更多同步/多缓冲 scope 参数，因此工程文档应区分“常规搜索项”和“只有 profile 指向特定 pass/同步问题时才试的高级项”。[证据 A：当前编译选项参考](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/environment_variable_and_compiler_options_reference.md#compiler-option-reference-table)

### 8.2 Ascend 扩展 API 的用途

当前案例使用 `triton.language.extra.cann.extension` 下的 `insert_slice` / `extract_slice` 在 UB 中拼接或切出子块，可把低维离散、高维连续的 GM 访问重组成连续 load + 片上转置。它们适合布局重排，不保证自动更快：重复插片本身可能拉长 Scalar/Vector 流水，必须 profile。

当前主线架构文档还列出了 Ascend affinity operators，如 `index_select`、`index_put`、`gather_out_to_ub`、`scatter_ub_to_out`、`indirect_load`、`indirect_store`，以及 `tl.compile_hint` 和跨 block 同步接口。这些属于更强的后端特化手段，应按 SoC/API 支持情况使用，不要为了“更像底层指令”而默认替换普通 Triton 表达式。[证据 A：架构与核心特性](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/architecture_design_and_core_features.md)

### 8.3 Ascend 950：把 SIMT / 间接访存作为独立优化路径

950 不应只在 compiler-option 表里出现一次。当前主线 `compile_mode` 提供：

- `simd`：纯 SIMD；
- `unstructured_in_simt`：混合模式，结构化部分走 SIMD，离散/非结构化访问尽量走 SIMT；
- `simt_only`：纯 SIMT。

同时，官方 API/示例已经出现 `gather_out_to_ub`、`indirect_load/store` 等更偏 950/新架构的访问手段；例如当前 `gather_out_to_ub` 示例在测试中明确限定为 Ascend 950。[证据 A：`gather_out_to_ub` 示例](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/zh/python-api/_examples/triton.language.extra.cann.extension.gather_out_to_ub.py)

因此 950 上的离散访问应比较三类方案：

```text
SIMD + UB regularization
vs
hybrid SIMT indirect access
vs
pure SIMT
```

A2/A3 上“尽量先规则化成连续 DMA”的经验仍很重要，但不能直接写成 950 的硬规则。

### 8.4 版本纪律

- 旧 `Ascend/triton-ascend` 仓库已经声明迁移到 `triton-lang/triton-ascend`；新代码优先以新主仓库文档/API 为准。[证据 A：迁移声明](https://github.com/Ascend/triton-ascend)
- 所有 compiler option、默认值和扩展 API 都记录 Triton-Ascend commit/版本。文档中 `main` 链接仅用于浏览，复现实验使用 commit permalink。
- 调试开关不要带入性能环境：强制重编译、IR dump、line info、interpreter、device print 都会改变编译或执行开销。
- `DISABLE_LLVM_OPT` 之类开关用于定位编译器问题，不是常规“优化按钮”；只有 A/B profile 证明目标 pass 在特定 kernel 上有回退时才保留局部 workaround。

## 9. 已有仓库中的优化案例

| 仓库/文件 | 算子 | 具体优化点 | 适用条件 | 证据等级 |
|---|---|---|---|---|
| [`triton-ascend` profiling](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/zh/debug_guide/profiling.md#示例i64i32-的compare在npu上无法启用vector导致向量计算转为标量计算) | LayerNorm | `tl.where` 中 i64 compare 改为 FP32 compare，使示例路径由 Scalar compare 转为 Vector cast/compare | 转换不改变比较语义；load/store mask 通常无需照搬 | A |
| [`triton-ascend-ops/003-ub_overflow`](https://github.com/Ascend/triton-ascend-ops/blob/755cf18c30f18720f67b6360c2b2856b64739822/tutorial/basic/003-ub_overflow.zh.md) | SGLang KV slot 分配简化版 | 长 `max_num_extend_tokens` 不一次搬入；按 `BLOCK_SIZE` 在核内循环 load/store | 长序列、一次 tile 会 UB overflow | A |
| [`triton-ascend-ops/004-discrete_memory_access`](https://github.com/Ascend/triton-ascend-ops/blob/755cf18c30f18720f67b6360c2b2856b64739822/tutorial/basic/004-discrete_memory_access.zh.md) | `out = x[idx]` | 连续加载候选 `x` 到 UB，再 `tl.gather` 片上选择，替代 GM 逐索引 load | 候选 `M` 能驻留 UB；预取额外字节可接受 | A |
| [`triton-ascend-ops/005-load_order`](https://github.com/Ascend/triton-ascend-ops/blob/755cf18c30f18720f67b6360c2b2856b64739822/tutorial/basic/005-load_order.zh.md) | 循环读 A/B、写 O/B | 把无依赖 `load A` 放到受循环携带依赖的 `load B` 前，使其可与上轮 `store B` 重叠 | 确认 A/B 无别名且重排不改变可见性 | A |
| [`triton-ascend-ops/006-tiling`](https://github.com/Ascend/triton-ascend-ops/blob/755cf18c30f18720f67b6360c2b2856b64739822/tutorial/basic/006-tiling.zh.md) | `gather(dim=1)` | 从 `(B, ceil(K/BK))` 大 grid 改为 batch 分块的一维 grid，并在每核循环 K tile | logical grid 远大于物理核；任务可独立分配 | A |
| [`triton-ascend-ops/002-decode_grouped_attention`](https://github.com/Ascend/triton-ascend-ops/blob/755cf18c30f18720f67b6360c2b2856b64739822/tutorial/best_practice/002-decode_grouped_attention.py) | Decode grouped attention | 对“低维离散、高维连续”的 K cache 按连续高维逐行 load，在 UB 用 `insert_slice` 拼接再转置；V 的高维离散/低维连续保持向量 load；同时做 online softmax + QK/PV | KV cache 索引离散；插片/循环开销低于直接二维离散 load | A（代码），B（收益需复测） |
| [`triton-ascend-ops/003-fused-cat-slice-conv1d`](https://github.com/Ascend/triton-ascend-ops/blob/755cf18c30f18720f67b6360c2b2856b64739822/tutorial/best_practice/003-fused-cat-slice-conv1d.zh.md) | Qwen3-Next causal conv1d update | 用 UB `insert_slice` 实现 cat，消除负 offset + `tl.where`；转置/“借轴转置”规避短尾轴 padding；grid 限制到 Vector Core 附近 | 文档限定特定简化分支；借轴技巧要求总字节对齐；shape/版本敏感 | A（机制），B（仓库报告 Atlas A3 1400→12 µs，未独立复现） |
| [`vllm-ascend/ops/triton/rope.py`](https://github.com/vllm-project/vllm-ascend/blob/c13036c9ca49ba2c5bb3a66247ce1efd1aaeb289/vllm_ascend/ops/triton/rope.py) | RoPE | 按 Vector Core 数设置 grid；token 维在 kernel 内跨步循环；head 维使用 `BLOCK_SIZE_HEAD`，大 head dim 时主动缩小 tile 以规避 UB 压力；Q/K 与 cos/sin 保持连续访问 | decode/prefill token 数变化大；head dim 较大或尾块明显 | A（代码），B（收益需复测） |
| [`vllm-ascend/ops/triton/rms_norm.py`](https://github.com/vllm-project/vllm-ascend/blob/c13036c9ca49ba2c5bb3a66247ce1efd1aaeb289/vllm_ascend/ops/triton/rms_norm.py) | RMSNorm | `do_not_specialize` 避免把动态 batch 参数制造成无谓编译变体；grid 贴近物理 Vector Core；每核循环 row block；按维度选择 2 的幂 `BLOCK_M` 平衡寄存器/UB | batch/token 动态且 kernel 高频调用 | A（代码） |
| [`vllm-ascend/ops/triton/bincount.py`](https://github.com/vllm-project/vllm-ascend/blob/c13036c9ca49ba2c5bb3a66247ce1efd1aaeb289/vllm_ascend/ops/triton/bincount.py) | bincount | 1D grid-stride loop 避免 65535 限制；`grid_size=min(core_num,total_blocks)`；原子/不规则访问关闭 `multibuffer` 候选；动态长度用 `do_not_specialize` | 输出 bins 大、输入长度动态、原子写入占主导 | A（代码） |
| [`vllm-ascend/ops/triton/batch_memcpy.py`](https://github.com/vllm-project/vllm-ascend/blob/c13036c9ca49ba2c5bb3a66247ce1efd1aaeb289/vllm_ascend/ops/triton/batch_memcpy.py) | batch memcpy | 循环外完成 pointer cast；流式读使用 `.cg` cache modifier；块循环复用地址计算，减少每次迭代的标量开销 | 纯搬运/流式访问且输入不会被相邻 tile 重用 | A（代码） |
| [`vllm-ascend` release notes](https://github.com/vllm-project/vllm-ascend/releases) | 多个 Triton kernel | 公开发布记录包含 rope、bincount、penalty、temperature/top-k/min-p 等 Triton kernel 优化，以及“减少参数触发的重复编译”；可作为优化候选清单，但具体收益必须按版本重测 | 线上服务中编译/dispatch 或采样 kernel 成为长板 | B（发布说明，未逐项独立复现） |
| [`FlagGems` Ascend backend](https://github.com/flagos-ai/FlagGems/tree/0b69247b68dbf0f43cd7c573b5b5cadb06d14e8d/src/flag_gems/runtime/backend/_ascend) | softmax/gather/index-select 等 | 后端 heuristic 按 `M/N/K` 与 tile 可整除性选择 block；softmax 根据 `K`、tile 和 AIV core 数估算 wave 数；小/中/大 N 分档选择 block 与并行度。注意仓库中仍有硬编码 core 数的 FIXME，应在产品代码改为设备属性查询 | 算子 shape 分布稳定、需要减少过度编译；硬件代际变化时必须刷新 heuristic | A（代码），C（收益待实测） |
| [`FlagGems` README](https://github.com/flagos-ai/FlagGems) | 多后端算子库 | 选择性手工优化、按函数运行时 dispatch、多后端后端分层；对 NPU 的落地应将“通用 Triton kernel”和 `_ascend` 特化 heuristic 分开测，避免把其他后端配置直接复制 | 需要维护通用实现，同时为 Ascend 保留 shape/硬件特化 | B（仓库说明） |
| [SGL Kernel NPU PR #454](https://github.com/sgl-project/sgl-kernel-npu/pull/454) | Qwen3.5 GDN / causal_conv1d prefill | 用 CPU scheduler metadata 替代 `[-1]`/`max()`/`any()` 的 device scalar 回读；支持预构建 chunk metadata | varlen/prefill、同一 metadata 跨层复用 | B（PR 机制明确，需目标版本复测） |
| [vLLM-Ascend PR #14066](https://github.com/vllm-project/vllm-ascend/pull/14066) | spec decode 输入 copy | 从 `grid=1 + batch 内循环` 改为 `grid=batch_size`，把独立 request 暴露给 Vector Core | 原实现并行度不足；batch 中 request 独立 | B（PR 报告 +47% throughput） |
| [vLLM-Ascend PR #13340](https://github.com/vllm-project/vllm-ascend/pull/13340) | multi-group slot mapping | 2D grid 一次并行全部 group；pointer/stride/block 参数在 init 阶段预构建并复用 | static group metadata、多 step 高频调用 | B（PR 报告旧参数构造约 5 ms/call） |
| [vLLM-Ascend PR #14499](https://github.com/vllm-project/vllm-ascend/pull/14499) | graph padding | 把数千次 scalar padding store 改为 `tl.arange + mask` 块向量 store | active batch 小、graph buffer 大、padding 长尾 | B（PR 报告约 1140→43 µs） |
| [FLA PR #1126](https://github.com/fla-org/flash-linear-attention/pull/1126) | causal_conv1d | 1D Vector core-grid；bulk/tail 用 `TAIL_MODE: constexpr` 拆编译路径，让 DCE 删除不用的 DMA buffer；stride-1 layout | runtime branch 同时撑大 UB live set；packed training | B（已合入，带 benchmark） |
| [FLA PR #1149](https://github.com/fla-org/flash-linear-attention/pull/1149) | KDA WY backward | 1D Vector core task-loop、T-contiguous host transpose、workspace 消除、`dA_acc` 留 UB、bulk/tail constexpr、pointer rebinding | launch/MTE/HBM round-trip/UB 都是瓶颈 | B（已合入，带 benchmark） |
| [FLA PR #1130](https://github.com/fla-org/flash-linear-attention/pull/1130) | KDA intra backward | 按阶段/UB 峰值拆 kernel，分别扩大 BC/BK；1D AICore task-loop | monolithic kernel 各阶段 live set 不同 | B（真实 Ascend 优化） |
| [FLA PR #1044](https://github.com/fla-org/flash-linear-attention/pull/1044) | fused LayerNorm/RMSNorm + gate | BT-tiled row 并行、按 forward/backward 不同 UB 预算选 tile、提前 store grad 缩短 live-range | row-serial、权重重复加载、UB 峰值限制 | B（报告约 14.9→6.0 ms） |
| [FLA PR #1132](https://github.com/fla-org/flash-linear-attention/pull/1132) | GDN chunk metadata | 允许 caller 传预计算 `chunk_indices`，避免每层重复 prepare | fixed/reused `cu_seqlens` | B（issue reporter 报约 25% 项目收益） |
| [FLA PR #1113](https://github.com/fla-org/flash-linear-attention/pull/1113) | 多个 Triton-Ascend `tl.dot` kernel | 对 lhs clobber 做重新 load/保留 pristine copy，防止 CUDA 语义假设导致静默错误 | 同一 lhs tile 被多个 dot 重复使用 | B（correctness hazard，不是通用性能规则） |
| [`MindSpeed-Ops` README](https://gitcode.com/Ascend/MindSpeed-Ops/blob/master/README.md) | 训练业务自定义算子（含 Triton/TileLang） | 仓库明确按 CANN/PyTorch/triton-ascend 版本配套，并要求用 ATK 分离 performance_device、accuracy 测试；可借鉴“同一算子固定输入生成、设备性能与精度分开验收”的工程流程 | 训练算子需要跨版本交付，且要同时维护 Triton 与其他后端实现 | B（仓库文档/测试流程） |
| [`MindSpeed-Ops tests`](https://gitcode.com/Ascend/MindSpeed-Ops/tree/master/tests) | Sinkhorn、RMS/Add 等 | 测试命令同时覆盖 Triton backend 与 NPU backend，并产出性能和精度结果；把这一流程迁移到新 kernel，可防止只优化设备时间却引入布局/Host 侧回退 | 需要比较 Triton、Ascend native 和端到端调用 | B（测试说明） |

## 10. 常见反模式和失败原因

| 反模式 | 常见症状 | 修正方向 |
|---|---|---|
| 把 GPU 式超大 logical grid 直接搬到 NPU | Block Dim 远大于物理核、启动轮次多、短 kernel 反而变慢 | 查询 Vector/AI Core 数；用物理核 grid + 核内 stride loop；只有无顺序依赖时才启用 AutoBlockify |
| 反过来把所有任务都塞进 1 个/少数 program | 大量独立 batch/row 串行，物理核空闲 | 把独立任务提升到 grid 维；并行度先填满有效核，再考虑核内循环 |
| 在 wrapper 里从 NPU tensor 取 scalar | Host gap、stream stall、graph capture 失败 | 使用 scheduler 已知的 CPU metadata；预构建并复用 chunk/block metadata |
| 运行时 branch 包含两套大 DMA/临时路径 | 实际只走一支但 UB 峰值仍很高，tile 放不大 | 把结构模式改成 `tl.constexpr`/Host dispatch，让 DCE 删除未用路径 |
| 一味 fusion | HBM 往返减少但 UB overflow、tile 被迫变小、编译变慢 | 按 peak live set 决定 split/fuse；不同阶段允许不同 BC/BK/BV |
| 长 padding/fill 用 scalar `for` 逐元素处理 | Scalar/FLOWCTRL 长尾，尤其 graph buffer 很大时 | 改成块向量 `tl.arange + mask + load/store` |
| 每层/每 step 重建稳定 metadata/descriptor | kernel 很快但 Python/tensor construction 占端到端大头 | init/shape-bucket 阶段预构建，按生命周期正确复用 |
| 在热循环中 `ptr +=` 维护地址状态 | 标量依赖链、编译器难识别 affine access，特定版本可能错误 lowering | 每轮从 base + task_id/loop_id 重新计算 local pointer |
| 默认认为 Ascend `tl.dot` 后 lhs 可像 CUDA 一样重复使用 | 结果静默漂移、只有部分 shape 出错 | 对目标版本加 repeated-dot correctness test；必要时保留 copy 或 reload |
| 只按输入/输出估 UB | 编译报 UB overflow，或打开 multibuffer 后突然溢出 | 把 mask、index、FP32 accumulator、布局临时量和 buffer 倍数计入峰值活跃集，保留编译器余量 |
| 逐索引从 GM 读取离散数据 | MTE 指令多、有效带宽低、Vector 空洞 | 评估连续候选区预取到 UB + `tl.gather`；若候选区过大或复用低则不要盲目预取 |
| 用负 offset + `tl.where` 模拟 cat/slice | Scalar/FLOWCTRL 上升，尾轴 padding 浪费 | 采用 `insert_slice`/片上重排；或通过轴交换使搬运连续并满足 32B/512B 对齐 |
| 为每个归约统计量重复读 GM | MTE2/MTE3 变长，端到端 kernel 数增加 | 工作集能驻留片上时做单 pass/online 算法；融合前先估算活跃集和数值稳定性 |
| 把所有整数表达式降成 int32 | 大索引 shape 出现越界或精度错误 | 只对已证明范围安全的 offset/compare 降型；FP32 compare 也要确认大整数可精确表示 |
| 盲调 `BLOCK_SIZE`、`num_warps` 或全开编译选项 | 某一 shape 变快，其他 shape/版本回退；编译缓存爆炸 | 以 profiler 瓶颈为依据，按 shape bucket autotune，并记录 SoC/CANN/Triton commit |
| 把调试开关带入性能数据 | 反复编译、line info/print/interpreter 造成额外耗时 | 调试与性能环境分离；只保留正式版本的 warmup、同步和固定频率测量 |
| 只看 kernel 时间、不看端到端 | kernel 加速但 `contiguous/transpose` 或 Host sync 抵消收益 | 同时报告设备 kernel、布局转换、dispatch、同步和总调用时间 |

这些反模式是从官方指南、`triton-ascend-ops` 教程和 vLLM-Ascend 实现归纳出的检查项；具体算子仍需通过 `msprof op` 和正确性测试升级证据等级。

## 11. Benchmark 与验收模板

每个优化候选至少保留一份可复现记录：

```text
硬件/软件：SoC、CANN、驱动、torch、torch_npu、triton-ascend commit
算子契约：输入 dtype、shape、stride、布局、随机种子、边界/非对齐样例
编译配置：BLOCK_M/N/K、BLOCK_SIZE_SUB、num_stages、multibuffer、NPU 选项
基线/候选：同一输入、同一输出语义、同一缓存清理与 warmup 规则
计时：warmup 次数、steady-state 次数、同步点、median/p95/p99
拆分：compile、dispatch、layout conversion、metadata build、Host↔Device sync、device kernel、端到端
JIT：variant 数、cache hit、首次/新增 shape 编译峰值、`do_not_specialize`/constexpr 策略
Graph：是否 capture 成功；是否因 device scalar / Python control-flow / dynamic allocation graph break
Host metadata：哪些值来自 CPU scheduler，哪些仍从 device 回读；哪些对象跨层/跨 step 复用
正确性：相对/绝对误差、NaN/Inf/极值、尾块、空输入和动态 shape
证据：kernel 名、msprof 命令、op_summary/PipeUtilization 文件路径
```

推荐流程：先跑 PyTorch/native NPU 参考和 Triton 基线，再做单变量改动；每个 shape bucket 至少重复多轮，先过滤精度/UB/编译失败配置，再比较性能。设备分析可用：

```bash
msprof op --kernel-name=<kernel_name> python benchmark.py
```

需要源码行号映射时，在重新编译前设置 `TRITON_DISABLE_LINE_INFO=0`。报告中明确 speedup 定义（例如 `baseline / candidate`），不要把冷启动编译时间与稳态 kernel 混在同一个数字中。

## 12. 按收益/风险排序的落地清单

**P0：测量 + 先消灭隐藏同步。** 建立 correctness golden、shape bucket、warmup/同步规范；记录 SoC/CANN/Triton 版本；同时排查 `.item/.cpu/.tolist/.any/.max`、metadata 重建、graph break、JIT compile spike。很多 serving 路径里这些问题比单个 kernel 更大。

**P1：低风险高复用。** 双向调整 grid：并行度不足时把 batch/row 暴露到 grid，logical tile 过细时再压回物理核附近；修复连续/对齐访问；向量化长 scalar loop；共享 invariant load；bin/group/pad 规则化离散访问；预构建稳定 metadata；缩短明显的 UB live-range。

**P2：改变 kernel dataflow。** 对 Scalar/FLOWCTRL 做索引/compare/address dependency 审查；对 MTE 重排 load、合并事务；对 Cube 调 M/N/K；用 `tl.constexpr` 拆 bulk/tail/optional-state，让 DCE 降低 UB 峰值；按 live-range 决定 fusion 还是 split；对多 pass 算子尝试 online 算法；对非结构动态参数用 `do_not_specialize` 控 JIT 变体。

**P3：compiler/SoC 特化。** 联合搜索 `multibuffer` 及其 scope、CV balance、tile-mix、workspace multibuffer、同步相关 option、AutoBlockify；Ascend 950 单独比较 SIMD、hybrid SIMT、SIMT-only 与专用 indirect/gather 路径。只在目标 SoC、软件栈和代表性 shape 上验证后上线，并保留回退配置。

上线门槛：端到端收益为正、所有精度/边界测试通过、无新增 UB/L1 溢出、编译缓存规模可接受、profile 中瓶颈确实转移或缩短。仓库报告的单点 speedup（例如教程中的 1400→12 µs）只能作为候选线索，不能替代本地复测。

## 13. 来源索引

仅收录实际用于本文结论的一手来源；检索入口、二手博客和无法复核的宣传数据不作为关键证据。

- [triton-lang/triton-ascend 编程指南](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/zh/programming_guide/index.md)
- [triton-lang/triton-ascend profiling 指南](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/zh/debug_guide/profiling.md)
- [GPU→NPU 迁移指南](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/en/migration_guide/migrate_from_gpu.md)
- [环境变量与编译选项](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/en/environment_variable_and_compiler_options_reference.md)
- [Vector/Cube/CV 编程指南](https://github.com/triton-lang/triton-ascend/tree/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/zh/programming_guide)
- [Ascend/triton-ascend-ops 教程](https://github.com/Ascend/triton-ascend-ops/tree/755cf18c30f18720f67b6360c2b2856b64739822/tutorial)
- [vLLM-Ascend Triton kernels](https://github.com/vllm-project/vllm-ascend/tree/c13036c9ca49ba2c5bb3a66247ce1efd1aaeb289/vllm_ascend/ops/triton)
- [vLLM-Ascend release notes](https://github.com/vllm-project/vllm-ascend/releases)
- [FlagGems Ascend backend](https://github.com/flagos-ai/FlagGems/tree/0b69247b68dbf0f43cd7c573b5b5cadb06d14e8d/src/flag_gems/runtime/backend/_ascend)
- [FlagGems 主仓库](https://github.com/flagos-ai/FlagGems)
- [MindSpeed-Ops README](https://gitcode.com/Ascend/MindSpeed-Ops/blob/master/README.md)
- [MindSpeed-Ops tests](https://gitcode.com/Ascend/MindSpeed-Ops/tree/master/tests)
- [SGL Kernel NPU PR #454：prebuilt metadata / D2H sync elimination](https://github.com/sgl-project/sgl-kernel-npu/pull/454)
- [vLLM-Ascend PR #14066：spec-decode batch parallelization](https://github.com/vllm-project/vllm-ascend/pull/14066)
- [vLLM-Ascend PR #13340：fused slot mapping + pre-cached params](https://github.com/vllm-project/vllm-ascend/pull/13340)
- [vLLM-Ascend PR #14499：vectorized graph padding](https://github.com/vllm-project/vllm-ascend/pull/14499)
- [FLA PR #1126：constexpr bulk/tail DMA split](https://github.com/fla-org/flash-linear-attention/pull/1126)
- [FLA PR #1149：KDA WY backward optimization](https://github.com/fla-org/flash-linear-attention/pull/1149)
- [FLA PR #1130：KDA intra backward split by UB peak](https://github.com/fla-org/flash-linear-attention/pull/1130)
- [FLA PR #1044：BT-tiled fused norm/gate](https://github.com/fla-org/flash-linear-attention/pull/1044)
- [FLA PR #1132：pre-computed chunk_indices](https://github.com/fla-org/flash-linear-attention/pull/1132)
- [FLA PR #1113：Ascend `tl.dot` lhs reuse hazard](https://github.com/fla-org/flash-linear-attention/pull/1113)
- [Triton-Ascend 架构与 Ascend affinity operators](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/architecture_design_and_core_features.md)
- [Triton-Ascend `gather_out_to_ub` 示例](https://github.com/triton-lang/triton-ascend/blob/5cdbf25baa17a5a64dcad1fa83ed54a62cbba414/docs/zh/python-api/_examples/triton.language.extra.cann.extension.gather_out_to_ub.py)
