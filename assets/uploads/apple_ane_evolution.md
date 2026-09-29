

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

## NPU (ANE) Evolution

![Geekbench AI](/Users/bytedance/Library/Application Support/typora-user-images/A20-preview-Geekbench-AI-NPU.jpg)

// AGENT: Merge M chip with A chip  columns like above

| M 系列        | 同期/同代 A 系列  | ANE              | 官方峰值/变化                          | 我认为的架构意义                                             |
| ------------- | ----------------- | ---------------- | -------------------------------------- | ------------------------------------------------------------ |
| **M1 (2020)** | **A14**           | 16-core          | **11 TFLOPS**                          | Mac 首次引入 ANE；建立 16-core 基线                          |
| **M2 (2022)** | A15/A16 时期      | 16-core          | **16 TFLOPS**                          | 主要是频率/执行效率提升，没有明显组织重构 [apple.com](https://www.apple.com/newsroom/2022/06/apple-unveils-m2-with-breakthrough-performance-and-capabilities/?utm_source=chatgpt.com) |
| **M3 (2023)** | **A17 Pro**       | 16-core          | **19 TFLOPS**                          | 3nm、更高性能/效率，仍是传统16-core ANE [apple.com](https://www.apple.com/li/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/?utm_source=chatgpt.com) |
| **M4 (2024)** | **A18 / A18 Pro** | 16-core          | int8: **38 TOPS**   fp16: **19TFLOPS** | 支持int8 x int8快速路径；仍保持16-core组织 [apple.com](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/?utm_source=chatgpt.com) // AGENT: cite this https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615?utm_source=chatgpt.com |
| **M5 (2025)** | **A19 / A19 Pro** | 16-core          | int8: **50 TOPS** fp16: **25 TFLOPS**  | ANE继续增强，但这一代 AI 架构重心明显开始转向 **GPU Neural Accelerator** [apple.com](https://www.apple.com/cf/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/?utm_source=chatgpt.com) |
| **M6 (2026)** | **A20 系列时期**  | **Dual 16-core** | int8: **100 TOPS** fp16: **50 TFLOPS** | **第一次明显横向扩展：2×16-core ANE，可被系统同时利用** [apple.com](https://www.apple.com/ca/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/?utm_source=chatgpt.com) |



### Inside the M5 ANE

> A step further from the great blogpost https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615?utm_source=chatgpt.com

#### Supported OPS

> copied from https://maderix.substack.com/p/inside-the-m4-apple-neural-engine?r=1afltp

![img](/Users/bytedance/Library/Application Support/typora-user-images/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0f8864de-4c70-453a-8d2b-d59f3ec9b9fd_663x86.png)

##### Supported OP Types (As Graph Input):

This is for shape\dtype validation, 

```
AffineTransform  ArgMinMax  BatchToSpace  Broadcast  ChannelToSpace  Concat

Conv  CropResize  CrossCorrelation  CrossProduct  Dropout  DynamicGOC

DynamicSlice  ElementWise  Flatten  Gather  GlobalArgMinMax  GOC

InputView  InstanceNorm  L2Norm  LayerNorm  Linear  LRN  MatrixMult

MinMaxNorm  Neuron  NMS  Pad  PixelShuffle  PixelUnshuffle  Pool  Random

Reduction  Resample  Reshape  Resize  ResizeAs  RingBufferWriter

ScaledElementWise  SDPA  Shape  Softmax  Sort  SpaceToBatch  SpaceToChannel

Tile  TopK  Transpose  Unflatten
```



##### High level dialect `anec.*` 

| 类别 | 算子 |
| --- | --- |
| 卷积/矩阵 | `convolution` `deconvolution` `linear` `matmul` `gain_offset_control`(GOC) `sdpa` |
| 归一化 | `batch_norm` `instance_norm` `layer_norm` `l2_norm` |
| 池化/规约 | `average_pool` `max_pool` `l2norm_pool` `reduce_avg` `reduce_max` `reduce_min` `reduce_sum` `arg_min_max` `global_arg_min_max` |
| 激活/一元 | `relu` `n_relu` `leaky_relu` `clamped_relu` `elu` `gelu` `swish` `sigmoid` `high_precision_sigmoid` `tanh` `erf` `sign` `abs` `sqrt` `r_sqrt` `sqr`/`square` `exp2` `log2` `sin` `cos` `ceil` `floor` `round_nearest` `trunc` `invert` |
| 逐元素二元 | `add` `sub` `mult` `div` `max` `min` `power` `scaled_elementwise` |
| 比较（出 mask） | `equal` `not_equal` `greater_than` `greater_than_equal` `less_than` `less_than_equal` + `*_zero` 系列 |
| 形状/布局 | `reshape` `flatten` `unflatten` `transpose` `concat` `broadcast` `tile` `padding` `crop_resize` `resize` `resample` `pixel_shuffle` `pixel_unshuffle` `space_to_batch` `batch_to_space` `space_to_channel` `channel_to_space` `input_view` `gather_nd` |
| 量化/精度 | `quant` `dequant` `cast` |
| 其它/框架 | `softmax` `dirac` `degamma` `state` `ring_buffer_reader` `ring_buffer_writer` `tensor_to_tensor_buffer`(及逆) `region_return` `unrealized_conversion_cast` |

##### Low level dialect `ane.*`

高层 op 最终被 lower 到两类执行单元——**卷积引擎 (NE)** 与 **平面引擎 (PE)**：

// AGENT: Make it a table

```text
ne_conv  ne_matmul  ne_pool  ne_bypass        # NE：卷积/矩阵核心 MAC 阵列

pe_elementwise  pe_goc  pe_pool               # PE：逐元素 / GOC / 池化
resample  cross_product  cost_volume
matrix_decomposition  gamma  degamma
```

> 这层揭示 ANE 的硬件结构：核心是 `ne_matmul`/`ne_conv` 的 MAC 阵列，`pe_*` 处理逐元素/池化。
> 无法映射到 NE/PE 的算子会退回 CPU/GPU。



#### SRAM hierachy:

according to the dieshot,M4 has:

- 8MB SLC
- 2MB ANE Cache

There's no 32MB SRAM dedicated for M4 chip. 

// Fold this image by default

![img](/Users/bytedance/Library/Application Support/typora-user-images/M4-Layout.jpg)

In terms of Matmul, the big matmul is tiled into small tiles, so the actually working set (even for 2048x2048, 4096x4096) should fit well into the 2MB ANE cache & 8MB SLC cache.

For example, I can split the 4096x4096x4096 matmul into small tile like 64x4096x64, which occupies 2 x 2 x 64 x 4096 =1,048,576 MByte. Simple scan&repeat 64x64 times then you get the full results.

The root cause for this performance drop is bad tiling strategy on the CoreML software stacks.

![img](/Users/bytedance/Library/Application Support/typora-user-images/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd3712a31-d925-41e3-85ab-b905fd18d296_561x270.png)

#### Convolution > matrix Multiplication

Qualcomm's QNN has this bug too. Actually compiler can do a matmul to conv1d transformation pass at the very beginning. I have no idea why these two industry leaders share the same stupid bug.

Some numerical stability issues I guess?



#### Deep Graphs Fill the Pipeline

> Currently tested on M1's ANE (My laptop is MacBookPro-16inch with M1Pro + 32GB)

It's an interesting topic (probably the most one) in ANE. A quick conclusion is:

ANE utilization benefits from both depth (number of layers) and width (complexity of one layer)

A single layer which is big enough can only utilize the ANE at 30%

![image-20260904183339692](/Users/bytedance/Library/Application Support/typora-user-images/image-20260904183339692.png)

Increasing number of layers can also fully utilize the ANE

![image-20260904183426942](/Users/bytedance/Library/Application Support/typora-user-images/image-20260904183426942.png)



bigger the layer is, less layers it needs to get high utilization

![image-20260904183127321](/Users/bytedance/Library/Application Support/typora-user-images/image-20260904183127321.png)



#### INT8 != FP16, 38 TOPS is achieved

> Measure on my colleague's MacBookPro14-M5-Pro 48GB

FP16xFP16 Conv: **25TFLOPS**

int8xint8 Conv: **50 TOPS**

Int8xint8 doubles the peak throughput



#### One more thing: ANE alignment

> Tested on M1Pro

these alignments are done on-the-fly, since compiled binary size is not changed whether aligned or not

NOTE: Alignment is not the actual PE size, it only reflects the tile size preference.

kernel_size: 1x1 (no alignment)

Batch: 1 (no alignment)

H: 1 (no alignment)

![image-20260905094246992](/Users/bytedance/Library/Application Support/typora-user-images/image-20260905094246992.png)

W: 32 (on M5Pro)

> W 轴本身还有个 16 字节 DMA granule (=8 个 fp16)的 pad 对齐

However on M1Pro, seems like W=32 triggers bank conflicts (strange!)

> - 细结构(每 8 一档台阶)= 16 字节 DMA granule (8 个 fp16);
> - 孤立尖峰(每 32 一个)= 64 字节 bank-interleave 冲突 (32 个 fp16)

但是这里W也不是最后一维，为啥会触发bank冲突呢？

![image-20260905184313161](/Users/bytedance/Library/Application Support/typora-user-images/image-20260905184313161.png)

Cin/Cout \*: 32 (\*My Guess is Cin alignment, cause there should be one accumulation axis)

![image-20260904193247625](/Users/bytedance/Library/Application Support/typora-user-images/image-20260904193247625.png)

Banks:

ANE 的片上池是 64 bank × 16 字节交织、周期 64 字节(=32 个 fp16)。C 维通过"interleave 对齐/补齐"体现为量化台阶;W 维通过"行 stride bank 冲突"体现为 32k 处的孤立尖峰。两者根子是同一个 32(fp16)周期。

W 轴本身还有个 16 字节 DMA granule (=8 个 fp16)的 pad 对齐



// AGENT: complete the references below

References:

- https://developer.apple.com/videos/play/tech-talks/111375
- https://hubweb.cn/apple-silicon/chip-m/
- https://ane-guide.readthedocs.io/en/latest/part-1-machine/02-execution-model.html