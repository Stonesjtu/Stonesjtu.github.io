

// AGENT: translate into English

# SoC Evolution



Apple seems like to do a tick-tock update between CPU+ANE subsystem and GPU subsystem

// AGENT: 世代部分同时展示M和A系列

// AGENT： column顺序为 CPU、ANE、GPU，这样能直观看到CPU和ANE的同步更新节奏

// AGENT: 当代主要架构重心就变为 CPU / GPU tick tock表

| 世代   | CPU                        | GPU                | ANE                        | 当代主要架构重心             |
| ------ | -------------------------- | ------------------ | -------------------------- | ---------------------------- |
| **M1** | 新平台                     | 新平台             | 16c / 11T ops              | **全新基线**                 |
| **M2** | 小改                       | 小改 Apple8        | 15.8T ops                  | **全局 refinement**          |
| **M3** | 中等升级                   | **★ Apple9 大改**  | 小/中改                    | **GPU Tock**                 |
| **M4** | **★ CPU明显升级**          | Apple9 refinement  | **★ ANE大幅增强至38T ops** | **CPU + ANE Tock**           |
| **M5** | 中/大升级                  | **★ Apple10 大改** | refinement                 | **GPU/AI-GPU Tock**          |
| **M6** | **★ CPU complex重构/扩张** | Apple10 refinement | **★ Dual 16c ANE**         | **CPU + ANE / process Tock** |

## GPU Evolution

![A20 preview GB6 compute](/Users/bytedance/Library/Application Support/typora-user-images/A20-preview-GB6-compute.jpg)

// AGENT: A20,M6 should be of apple10 as predicted

| GPU Family  | 代表 SoC        | 架构定位                 | 最重要变化                                                   |
| ----------- | --------------- | ------------------------ | ------------------------------------------------------------ |
| **Apple7**  | A14 / M1        | 现代 Apple GPU 基线      | TBDR + SIMD32 + UMA；M 系列 GPU 起点                         |
| **Apple8**  | A15/A16 / M2    | Apple7 增强版            | cache/BW/效率增强，shader core 没有根本重构                  |
| **Apple9**  | A17/A18 / M3/M4 | **Dynamic GPU**          | 全新 shader core、**Dynamic Caching**、硬件 RT、**动态 occupancy** |
| **Apple10** | A19 / M5        | **AI heterogeneous GPU** | 每 GPU core 加 **Neural Accelerator**、FP16 增强 (变为2x FP32的吞吐）、Dynamic Caching Gen2 |

### Basic architecture 

GPU subsystem:

L2 cache --> GPU core --> 4x Shader Core --> 32 SIMD lanes --> ALU pipes

![image-20260903215304210](/Users/bytedance/Library/Application Support/typora-user-images/image-20260903215304210.png)

Shader Core (Streaming Multiprocessor / SM Core)

![image-20260903221420009](/Users/bytedance/Library/Application Support/typora-user-images/image-20260903221420009.png)

### Preliminary -- Occupancy

Higher occupancy leads to higher throughput, a.k.a. more TFLOPS achieved.

![image-20260909112302693](/Users/bytedance/Library/Application Support/typora-user-images/image-20260909112302693.png)

### Dynamic Caching

1. Dynamic register allocation
2. dynamic on-chip memory
6. Occupancy Manager：Decide when to schedule more SIMDGroups, when to schedule less to prevent register spilling

// AGENT: 2 column layout to compare before vs after

Before 1:

- Registers are pre-allocated before the simd group (warp) can be scheduled.
- The occupancy is limited by register pressure

![image-20260903222338675](/Users/bytedance/Library/Application Support/typora-user-images/image-20260903222338675.png)

After 1:

- Registers are allocated on demand
- two simd group can overlap, thus occupancy is double in this case
- Higher occupancy leads to more parallelism opportunities (like multiple issues, memory latency hidding)

![image-20260903222458222](/Users/bytedance/Library/Application Support/typora-user-images/image-20260903222458222.png)

// AGENT: 2 column layout to compare before vs after

Before 2：

- Register files, thread-group memory and buffer-stack cache are 3 independent memory regions
- Occupancy is limited by all the 3 resources
- Application should balance the usage of registers and thread-group memory, otherwise memory will be wasted and occupancy will be low leading to low TFLOPS

![image-20260908183940863](/Users/bytedance/Library/Application Support/typora-user-images/image-20260908183940863.png)

After 2:

- All resources are in a unified cache, dynamically partitioned and allocated
- Easily get high performance for various workloads

![image-20260908184054401](/Users/bytedance/Library/Application Support/typora-user-images/image-20260908184054401.png)



// AGENT: arrange the following 3 images into a single gif picture, with the animation of 3 blocks (registers/threadgroup memory/buffer adjusting the partition dynamically)

![image-20260908184233303](/Users/bytedance/Library/Application Support/typora-user-images/image-20260908184233303.png)

![image-20260908184200280](/Users/bytedance/Library/Application Support/typora-user-images/image-20260908184200280.png)

![image-20260908184134131](/Users/bytedance/Library/Application Support/typora-user-images/image-20260908184134131.png)

### Neural Accelerator (NAX)

> Apple GPU‘s Tensor Core era



![image-20260908190325128](/Users/bytedance/Library/Application Support/typora-user-images/image-20260908190325128.png)



The software stacks: TensorOps

- MatMul + Conv
- Same API across M1-M5
- Use NAX on M5(A19Pro)
- Can be mixed with other metal shader code (programmable)

![image-20260909110439856](/Users/bytedance/Library/Application Support/typora-user-images/image-20260909110439856.png)

Important features:

- 26.1: bf16 support

- 26.3: **cooperative tensors** as input (user-defined prologue for custom dequantization)

- 26.4: 4-bit / 8-bit int support

- 27: micro-scaling mxfp4 / mxfp8 / fp8 / 2-bit int

Cooperative tensors

// AGENT: 2 column layout to compare before vs after

Before

- Matmul output is written from and read-back to **slow** device memory

![image-20260909103005752](/Users/bytedance/Library/Application Support/typora-user-images/image-20260909103005752.png)

after

- Matmul output is written from and read-back to **fast** thread memory

![image-20260909103047334](/Users/bytedance/Library/Application Support/typora-user-images/image-20260909103047334.png)

Howto:

![image-20260909103203127](/Users/bytedance/Library/Application Support/typora-user-images/image-20260909103203127.png)

## Profiling&Debugging

### Model Level: Instrument

> Similar to NSYS

![image-20260909103856481](/Users/bytedance/Library/Application Support/typora-user-images/image-20260909103856481.png)

### Kernel Level: Xcode metal debugger

> Similar to NCU

![image-20260909103940068](/Users/bytedance/Library/Application Support/typora-user-images/image-20260909103940068.png)

// AGENT: complete the references below

References:

- https://developer.apple.com/videos/play/tech-talks/111375
- https://developer.apple.com/videos/play/tech-talks/111432
- https://hubweb.cn/apple-silicon/chip-m/
- https://ane-guide.readthedocs.io/en/latest/part-1-machine/02-execution-model.html