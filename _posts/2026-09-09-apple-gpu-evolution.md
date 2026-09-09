---
layout: post
title: "Apple GPU Evolution"
topic: "GPU architecture"
date: 2026-09-09
last_modified_at: 2026-09-09T19:25:58+08:00
excerpt: "Apple GPU evolution from Dynamic Caching to per-core Neural Accelerators and programmable TensorOps."
---

## SoC Evolution

Apple appears to follow a tick-tock cadence between the CPU + ANE subsystem[^ane-execution] and the GPU subsystem.


| Generation | CPU | ANE | GPU | CPU / GPU tick-tock |
| --- | --- | --- | --- | --- |
| **M1 / A14** | New platform | 16 cores / 11 TOPS | New Apple7 platform | **New baseline** |
| **M2 / A15–A16** | Minor update | 15.8 TOPS | Minor Apple8 update | **Overall refinement** |
| **M3 / A17 Pro** | Medium update | Minor / medium update | **Major Apple9 update** | <span class="tock-badge tock-badge--gpu">GPU tock</span> |
| **M4 / A18** | **Major CPU update** | **Major increase to 38 TOPS** | Apple9 refinement | <span class="tock-badge tock-badge--cpu">CPU/ANE tock</span> |
| **M5 / A19** | Medium / major update | M5 Max: **25 FP16 TFLOPS / 50 INT8 TOPS (measured)**[^m5-max-ane-measured]; A19: refinement | **Major Apple10 update** | <span class="tock-badge tock-badge--gpu">GPU tock</span> |
| **M6 / A20 (predicted)** | **CPU complex redesign / expansion** | **Dual 16-core ANE** | Apple10 refinement | <span class="tock-badge tock-badge--cpu">CPU/ANE tock</span> |

## GPU Evolution

> The A20 and M6 GPUs are predicted here to remain in the Apple10 family.

<figure class="post-figure">
  <img src="{{ '/assets/apple-gpu-evolution/a20-preview-gb6-compute.jpg' | relative_url }}" alt="Preview chart of A20 Geekbench 6 compute performance." />
  <figcaption><strong>A20 preview.</strong> This forecast is supporting context for the predicted row, not a measured or official result.</figcaption>
</figure>

| GPU family | Representative SoCs | Peak FP32 TFLOPS (total / per core) | Most important change |
| --- | --- | --- | --- |
| <strong class="gpu-family-label">Apple7</strong> | A14 / M1 | A14: **0.65 / 0.16**<br>M1: **2.62 / 0.33** | TBDR + SIMD32 + UMA; starting point of the M-series GPU |
| <strong class="gpu-family-label">Apple8</strong> | A15 / A16 / M2 | A15: **1.71 / 0.34**<br>A16: **~1.79 / ~0.36**<br>M2: **3.58 / 0.36** | Cache, bandwidth, and efficiency improvements; no fundamental shader-core redesign |
| <strong class="gpu-family-label">Apple9</strong> | A17 / A18 / M3 / M4 | A17 Pro: **2.06 / 0.34**<br>A18 Pro: **2.26 / 0.38**<br>M3: **3.43 / 0.34**<br>M4: **4.04 / 0.40** | New shader core, **Dynamic Caching**, hardware ray tracing, and dynamic occupancy[^family9] |
| <strong class="gpu-family-label">Apple10</strong> | A19 / A20 (predicted) / M5 / M6 (predicted) | A19 Pro: **2.49 / 0.41**<br>M5: **4.15 / 0.41**<br>A20 / M6: **TBD** | A **Neural Accelerator** in every GPU core, 2× FP16 versus FP32 throughput, and Dynamic Caching Gen2[^m5-gpu] |

Each entry reads **total / per GPU core**. The values are theoretical FP32 vector peaks for the highest-core base A- or M-series configuration, not application benchmarks. Except for Apple's published 2.6-TFLOPS M1 figure, they are estimates from reported GPU width and clock rate, rounded to two decimals; the per-core value divides that total by the GPU core count. Pro, Max, and Ultra variants are not included.[^gpu-peak]

### Basic architecture

GPU subsystem:

`GPU subsystem → L2 cache → GPU core → 4× Shader Core → 32 SIMD lanes → ALU pipes`

<figure class="post-figure">
  <img src="{{ '/assets/apple-gpu-evolution/gpu-subsystem.png' | relative_url }}" alt="Apple GPU subsystem diagram from shared cache to GPU and shader cores." />
  <figcaption><strong>GPU subsystem.</strong> The high-level organization from cache to GPU cores and shader cores.</figcaption>
</figure>

Shader Core (Streaming Multiprocessor / SM Core):

<figure class="post-figure">
  <img src="{{ '/assets/apple-gpu-evolution/shader-core.png' | relative_url }}" alt="Apple shader-core diagram showing SIMD lanes and execution resources." />
  <figcaption><strong>Shader core.</strong> The execution block that schedules SIMDgroups across the core's pipelines.</figcaption>
</figure>

### Preliminary — Occupancy

Higher occupancy leads to higher throughput: more of the available TFLOPS can be achieved when more SIMDgroups are ready to use the execution pipelines.[^family9]

<figure class="post-figure">
  <img src="{{ '/assets/apple-gpu-evolution/occupancy.png' | relative_url }}" alt="Occupancy diagram illustrating more concurrent SIMDgroups filling execution slots." />
  <figcaption><strong>Occupancy.</strong> More resident SIMDgroups create more opportunities to hide latency and keep execution pipelines busy.</figcaption>
</figure>

### Dynamic Caching

1. Dynamic register allocation
2. Dynamic on-chip memory
3. Occupancy Manage (Not covered in this post): decides when to schedule more SIMDgroups and when to schedule fewer to prevent register spilling[^family9]

<div class="post-compare-grid">
  <figure class="post-figure">
    <img src="{{ '/assets/apple-gpu-evolution/registers-before.png' | relative_url }}" alt="Before Dynamic Caching, registers are allocated for peak SIMDgroup demand." />
    <figcaption><strong>Before.</strong> Registers are pre-allocated before a SIMDgroup can be scheduled, so occupancy is limited by peak register pressure.</figcaption>
  </figure>
  <figure class="post-figure">
    <img src="{{ '/assets/apple-gpu-evolution/registers-after.png' | relative_url }}" alt="With Dynamic Caching, registers are allocated on demand and SIMDgroups overlap." />
    <figcaption><strong>After.</strong> Registers are allocated on demand. Two SIMDgroups can overlap in this example, doubling occupancy and exposing more parallelism.</figcaption>
  </figure>
</div>

<div class="post-compare-grid">
  <figure class="post-figure">
    <img src="{{ '/assets/apple-gpu-evolution/memory-before.png' | relative_url }}" alt="Separate register file, threadgroup memory, and buffer-stack cache before unified on-chip memory." />
    <figcaption><strong>Before.</strong> Register files, threadgroup memory, and buffer-stack cache are independent regions. Occupancy is constrained by all three, and an imbalanced workload can leave capacity unused.</figcaption>
  </figure>
  <figure class="post-figure">
    <img src="{{ '/assets/apple-gpu-evolution/memory-after.png' | relative_url }}" alt="Unified on-chip cache dynamically partitioned among registers, threadgroup memory, and buffers." />
    <figcaption><strong>After.</strong> The resources share a unified cache that is dynamically partitioned and allocated for different workloads.</figcaption>
  </figure>
</div>

<figure class="post-figure">
  <img src="{{ '/assets/apple-gpu-evolution/dynamic-memory-partition.gif' | relative_url }}" alt="Animation of registers, threadgroup memory, and buffer cache changing their shares of unified on-chip memory." />
  <figcaption><strong>Dynamic partitioning.</strong> Registers, threadgroup memory, and buffer storage adjust their shares of the same on-chip capacity as workload demand changes.</figcaption>
</figure>

### Neural Accelerator (NAX)

> The Tensor Core era of the Apple GPU

M5 and A19 place dedicated matrix-multiplication hardware inside every shader core.[^m5-gpu]

<figure class="post-figure">
  <img src="{{ '/assets/apple-gpu-evolution/neural-accelerator.png' | relative_url }}" alt="Apple GPU core diagram with a Neural Accelerator alongside conventional shader pipelines." />
  <figcaption><strong>Neural Accelerator.</strong> Dedicated matrix-multiplication hardware sits alongside the conventional shader pipelines.</figcaption>
</figure>

#### TensorOps

- MatMul + Conv
- Same API across M1–M5
- Uses NAX on M5 and A19 Pro
- Can be mixed with other programmable Metal shader code[^m5-gpu]

<figure class="post-figure">
  <img src="{{ '/assets/apple-gpu-evolution/tensorops-stack.png' | relative_url }}" alt="Apple TensorOps software stack from frameworks to Metal shaders and Neural Accelerators." />
  <figcaption><strong>TensorOps stack.</strong> The same API falls back to optimized shaders on older GPUs and uses Neural Accelerators on M5 and A19.</figcaption>
</figure>

Important features:

| Version | Feature |
| --- | --- |
| **26.1** | BF16 support |
| **26.3** | **Cooperative tensors** as inputs, enabling a user-defined prologue for custom dequantization |
| **26.4** | 4-bit / 8-bit integer support |
| **27** | Micro-scaling MXFP4 / MXFP8 / FP8 / 2-bit integer support |

#### Cooperative tensors

Cooperative tensors keep a MatMul result in fast thread memory for in-place post-processing.[^m5-gpu]

<div class="post-compare-grid">
  <figure class="post-figure">
    <img src="{{ '/assets/apple-gpu-evolution/cooperative-before.png' | relative_url }}" alt="Before cooperative tensors, a matrix result travels through device memory before custom processing." />
    <figcaption><strong>Before.</strong> The MatMul output is written to and read back from slower device memory.</figcaption>
  </figure>
  <figure class="post-figure">
    <img src="{{ '/assets/apple-gpu-evolution/cooperative-after.png' | relative_url }}" alt="With cooperative tensors, a matrix result remains in per-thread on-chip memory for custom processing." />
    <figcaption><strong>After.</strong> The MatMul output stays in fast thread memory for in-place post-processing.</figcaption>
  </figure>
</div>

How to:

<figure class="post-figure">
  <img src="{{ '/assets/apple-gpu-evolution/cooperative-howto.png' | relative_url }}" alt="Metal code showing how to create and consume a cooperative tensor." />
  <figcaption><strong>Cooperative tensor API.</strong> The result is distributed across participating threads, modified in place, then stored once.</figcaption>
</figure>

## Profiling & Debugging

### Model level: Instruments

> Similar to Nsight Systems

Instruments shows CPU, GPU, memory, display, command-buffer execution, and performance counters on a system timeline.[^m5-gpu]

<figure class="post-figure">
  <img src="{{ '/assets/apple-gpu-evolution/instruments.png' | relative_url }}" alt="Instruments Metal System Trace showing GPU work on a timeline." />
  <figcaption><strong>Metal System Trace.</strong> A model-level view of GPU work on the system timeline.</figcaption>
</figure>

### Kernel level: Xcode Metal debugger

> Similar to Nsight Compute

GPU trace replay isolates work for cost graphs, counters, register use, divergence, and instruction analysis.[^m5-gpu]

<figure class="post-figure">
  <img src="{{ '/assets/apple-gpu-evolution/xcode-metal-debugger.png' | relative_url }}" alt="Xcode Metal debugger showing an isolated GPU capture and shader performance data." />
  <figcaption><strong>Xcode Metal debugger.</strong> A kernel-level view of an isolated GPU capture and shader performance data.</figcaption>
</figure>

## References

[^family9]: Apple Developer, [Explore GPU advancements in M3 and A17 Pro](https://developer.apple.com/videos/play/tech-talks/111375/), Tech Talks: Apple family 9 shader cores, Dynamic Caching, flexible on-chip memory, dynamic occupancy, hardware ray tracing, and mesh shading.
[^m5-gpu]: Apple Developer, [Accelerate your machine learning workloads with the M5 and A19 GPUs](https://developer.apple.com/videos/play/tech-talks/111432/), Tech Talks: per-shader-core Neural Accelerators, TensorOps, cooperative tensors, Metal System Trace, and Xcode Metal debugger.
[^m5-max-ane-measured]: Kaiyu Shi, local M5 Max ANE throughput measurement, September 9, 2026: 25 TFLOPS at FP16 and 50 TOPS at INT8; measured results rather than Apple-published peak specifications.
[^gpu-peak]: Apple, [Apple unleashes M1](https://www.apple.com/newsroom/2020/11/apple-unleashes-m1/), November 2020, reporting 2.6 TFLOPS for M1; Philip Turner, [Apple GPU microarchitecture benchmarks](https://github.com/philipturner/metal-benchmarks), FP32 FLOPS, core-count, and clock tables through Apple8; HubWeb, [A-series](https://hubweb.cn/apple-silicon/chip-a/) and [M-series](https://hubweb.cn/apple-silicon/chip-m/) comparison tables, used for later GPU width and clock inputs. Later values are derived estimates, not Apple-published peak specifications.
[^m-series]: HubWeb, [Apple M-series specification comparison](https://hubweb.cn/apple-silicon/chip-m/): secondary comparison source for M-series generation data.
[^ane-execution]: Apple Neural Engine: A Complete Guide, [Execution model](https://ane-guide.readthedocs.io/en/latest/part-1-machine/02-execution-model.html): reverse-engineered ANE execution context; not an Apple specification.
