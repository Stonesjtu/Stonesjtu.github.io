---
layout: post
title: "Kernel Dispatch on Apple GPUs"
description: "Metal and CUDA command submission, with a Qwen3-30B-A3B decode trace on MLX."
topic: "GPU execution"
date: 2026-11-17
---

## Overview

| Aspect | Apple GPU (Metal) | NVIDIA GPU (CUDA) |
| --- | --- | --- |
| Work ordering | Command queue (`MTLCommandQueue`)[^metal-commands] | Stream (`cudaStream_t`)[^cuda-async] |
| Submission | Encode command buffer → commit to queue[^metal-commands] | Enqueue kernels or launch a graph on a stream[^cuda-async] |
| Grouping and replay | Single-use command buffer<br />Reusable indirect commands[^metal-commands][^metal-icb] | CUDA Graph DAG → instantiate → replay[^cuda-async] |
| Kernel ordering | Concurrent compute pass<br />Memory barriers for dependencies[^metal-sync] | In-order stream by default<br />Overlap across streams or independent graph nodes[^cuda-async] |
| Cross-queue / stream sync | Event signal → wait[^metal-sync] | `cudaEventRecord()` → `cudaStreamWaitEvent()`[^cuda-async] |

Command Queue：

- intra-queue: CBs are executed in order of commit
- inter-queue: needs explicit barrier and event, fence

Command Buffer:

- Make -> **Encode** -> Commit -> Completion
- Atomic unit for GPU scheduling
- No Replay (<40 ops per buffer in MLX)
- Different tasks can be encoded into a single command buffer

Compute Encoder (there are other encoders for GPU tasks （blit、render))：

- AI inference is compute only, so only 1 compute encoder in command buffer. (MLX does so)

- Kernels are dispatched concurrently
- Dependencies are explicitly inserted as barrier

<figure class="post-figure">
  <img src="{{ '/assets/apple-gpu-kernel-issue/metal-command-structure.png' | relative_url }}" alt="Metal command queue pointing to a command buffer with three compute encoders, each containing kernel dispatches." />
  <figcaption>A Metal command buffer groups compute passes, and each compute encoder records kernel dispatches. <a href="{{ '/assets/apple-gpu-kernel-issue/metal-command-structure.png' | relative_url }}">Full-size image</a></figcaption>
</figure>



## Execution Model: Commit & Completion

<figure class="post-figure">
  <img src="{{ '/assets/apple-gpu-kernel-issue/commit-and-completion.png' | relative_url }}" alt="A command buffer is committed to the GPU, which signals completion through a handler on the CPU side." />
  <figcaption>Committing a command buffer submits work to the GPU. The completion handler runs after that work finishes. <a href="{{ '/assets/apple-gpu-kernel-issue/commit-and-completion.png' | relative_url }}">Full-size image</a></figcaption>
</figure>



## Multiple Command Queues & Command Buffers

<figure class="post-figure">
  <img src="{{ '/assets/apple-gpu-kernel-issue/queues-and-buffers.png' | relative_url }}" alt="Multiple Metal command queues and three command buffers, with separate arrows for enqueue and commit." />
  <figcaption>Enqueue reserves a command buffer’s position in its queue. Commit submits the buffer for execution. <a href="{{ '/assets/apple-gpu-kernel-issue/queues-and-buffers.png' | relative_url }}">Full-size image</a></figcaption>
</figure>



## Case Study: Qwen3-30B-A3B on MLX

**Observed command structure**

The trace captured 10 decode steps with a 12-layer toy model:

- 235 command buffers
- 199 with a compute encoder
- 36 synchronization or control buffers with no encoder
- Almost every nonempty command buffer had **one compute encoder**
- About 4–30 kernel dispatches per encoder
- About 20 compute command buffers per token on average

A typical sequence looks like this:

```text
CommandQueue
├─ CommandBuffer 0
│  └─ ComputeEncoder 0: 16 dispatches
├─ CommandBuffer 1
│  └─ ComputeEncoder 1: 30 dispatches
├─ CommandBuffer 2
│  └─ ComputeEncoder 2: 4 dispatches
├─ CommandBuffer 3
│  └─ ComputeEncoder 3: 28 dispatches
...
```

**A layer can span multiple command buffers**

In the trace, `cb1` contains:

```text
disp0–5    Tail of the previous layer’s MoE block
           divide → SwiGLU → down_proj → reduce → residual add

disp6–19   Attention in the next layer
           RMSNorm
           Q/K/V int4 GEMV
           RoPE
           KV cache copy
           SDPA passes 1 and 2
           o_proj
           residual add

disp20–29  First part of the next layer’s MoE block
           RMSNorm
           router GEMV
           softmax / top-k / gather
           expert gate_proj
```

The next buffers contain:

```text
cb2: 4 dispatches
  expert up_proj
  divide
  fused sigmoid/multiply
  expert down_proj

cb3:
  MoE reduce + residual
  start the next layer’s attention
```

The boundaries therefore fall roughly here:

```text
Layer N tail | Layer N+1 attention | First part of Layer N+1 MoE
------------------- command buffer boundary -------------------
Remaining dispatches of Layer N+1 MoE before reduce
------------------- command buffer boundary -------------------
Layer N+1 tail | Layer N+2 attention | ...
```

A boundary can fall inside a single MoE block.

**Why MLX splits work this way**

In the MLX source revision examined here, the M1 Pro uses these default thresholds:[^mlx-metal]

```text
MLX_MAX_OPS_PER_BUFFER = 40
MLX_MAX_MB_PER_BUFFER  = 40
```

The commit check in `device.cpp` is:[^mlx-commit]

```cpp
return (buffer_ops_ > max_ops) ||
       ((buffer_sizes_ >> 20) > max_mb);
```

- Each dispatch increments `buffer_ops_`.
- Binding an input buffer for the first time adds its size to `buffer_sizes_`.
- Quantized MoE expert weights are large, so the buffer often crosses the roughly 40 MB resource threshold first.
- MLX checks the thresholds after encoding a primitive, so a command buffer can run past 40 dispatches.

When the commit check fires, MLX:

```text
endEncoding()
commit()
create a new command buffer
continue encoding subsequent primitives
```

MLX creates its compute encoder with:[^mlx-metal]

```cpp
computeCommandEncoder(MTL::DispatchTypeConcurrent)
```

MLX tracks buffer reads and writes and inserts a `memoryBarrier` between dependent kernels in the same encoder.[^mlx-metal]

> In this trace, the Qwen3 quarter model uses about 20 compute command buffers per token, usually with one concurrent compute encoder each. Layers, attention, and MoE can all span command buffers. The roughly 40 MB resource budget is the main factor behind where those boundaries fall, and they can occur within individual model blocks.

## References

[^metal-commands]: Apple, [MTLCommandBuffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer), command encoding, queue order, and commit semantics.
[^metal-icb]: Apple, [Indirect command encoding](https://developer.apple.com/documentation/metal/indirect-command-encoding), reusable commands in indirect command buffers.
[^metal-sync]: Apple, [Resource synchronization](https://developer.apple.com/documentation/metal/resource-synchronization), concurrent compute passes, memory barriers, and event synchronization.
[^cuda-async]: NVIDIA, [CUDA Programming Guide §2.5, Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html), stream ordering, events, and graph capture and replay.
[^mlx-commit]: MLX, [`CommandEncoder::needs_commit()`](https://github.com/ml-explore/mlx/blob/65be04707afe945f1825c2fb507b70473cd91fd0/mlx/backend/metal/device.cpp#L511-L514), command-buffer commit check.
[^mlx-metal]: MLX, [Metal `device.cpp`](https://github.com/ml-explore/mlx/blob/65be04707afe945f1825c2fb507b70473cd91fd0/mlx/backend/metal/device.cpp), buffer accounting, concurrent compute encoding, memory barriers, and device defaults.
