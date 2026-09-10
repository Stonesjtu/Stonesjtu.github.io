---
layout: post
title: "Apple GPU Evolution"
topic: "GPU architecture"
date: 2026-09-09
last_modified_at: 2026-09-10T17:03:00+08:00
excerpt: "Apple GPU evolution from Dynamic Caching to Neural Accelerators, with an MLX Metal GEMM source walkthrough."
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

| GPU family | Representative SoCs | Peak SIMD-group FP16 TFLOPS (total / per core) | Peak NAX FP16 TFLOPS (total / per core) | Most important change |
| --- | --- | --- | --- | --- |
| <strong class="gpu-family-label">Apple7</strong> | A14 / M1 | A14: **0.65 / 0.16**<br>M1: **2.62 / 0.33** | — | TBDR + SIMD32 + UMA; starting point of the M-series GPU |
| <strong class="gpu-family-label">Apple8</strong> | A15 / A16 / M2 | A15: **1.71 / 0.34**<br>A16: **~1.79 / ~0.36**<br>M2: **3.58 / 0.36** | — | Cache, bandwidth, and efficiency improvements; no fundamental shader-core redesign |
| <strong class="gpu-family-label">Apple9</strong> | A17 / A18 / M3 / M4 | A17 Pro: **2.06 / 0.34**<br>A18 Pro: **2.26 / 0.38**<br>M3: **3.43 / 0.34**<br>M4: **4.04 / 0.40** | — | New shader core, **Dynamic Caching**, hardware ray tracing, and dynamic occupancy[^family9] |
| <strong class="gpu-family-label">Apple10</strong> | A19 / A20 (predicted) / M5 / M6 (predicted) | A19 Pro: **4.98 / 0.82**<br>M5: **8.30 / 0.82**<br>A20 / M6: **TBD** | A19 Pro: **~9.96 / ~1.64**<br>M5: **~16.60 / ~1.64**<br>A20 / M6: **TBD** | A **Neural Accelerator** in every GPU core, 2× FP16 versus FP32 throughput, and Dynamic Caching Gen2[^m5-gpu] |

Each entry reads **total / per GPU core** for the highest-core base A- or M-series configuration. For Apple7–9, this table treats peak FP16 throughput as equal to FP32. For Apple10, SIMD-group FP16 is calculated as 2× FP32, while NAX FP16 is estimated as approximately 2× SIMD-group FP16. The NAX values are derived estimates, not measurements or Apple-published peaks. Except for Apple's published 2.6-TFLOPS M1 figure, the base values come from reported GPU width and clock rate and are rounded to two decimals. Pro, Max, and Ultra variants are not included.[^gpu-peak]

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

#### Speedup from NAX

> NAX changes prompt processing far more than token generation.

The cleanest complete comparison in the `llama.cpp` reports uses the same 40-core M5 Max and Llama 2 7B workload: `pp512`, `tg128`, and full GPU offload. The legacy result is the mean of two independent `8e672ef` submissions, which agree within 1.4%; the NAX result is the accepted `c1d0e7a` Tensor API submission.[^llamacpp-m5-max]

| Phase | Format | Without NAX, legacy Metal (t/s) | With NAX, Tensor API (t/s) | Speedup |
| --- | --- | ---: | ---: | ---: |
| Prefill (`pp512`) | F16 | 1,025.24 | 3,158.49 | **3.08x** |
| Prefill (`pp512`) | Q8_0 | 1,053.21 | 3,143.81 | **2.98x** |
| Prefill (`pp512`) | Q4_0 | 988.82 | 3,219.99 | **3.26x** |
| Decode (`tg128`) | F16 | 37.82 | 37.11 | **0.98x** |
| Decode (`tg128`) | Q8_0 | 64.86 | 72.42 | **1.12x** |
| Decode (`tg128`) | Q4_0 | 103.48 | 119.92 | **1.16x** |

Prefill improves by roughly **3x**, while decode changes by **-2% to +16%**. This is the expected shape: prefill exposes large matrix multiplications to NAX, whereas single-token decode remains dominated by moving model weights through memory.

This is a strong field comparison, but not a controlled same-build toggle: `llama.cpp` also improved between the two commits. A same-device branch-versus-main test from the original Tensor API pull request provides a useful cross-check: Mistral 7B Q4_0 prefill rose from 252.82 to 608.05 t/s (**2.40x**), while decode stayed effectively flat at 27.55 versus 26.59 t/s (**0.97x**).[^llamacpp-nax-ab]

`Q8_0` and `Q4_0` are GGUF block-quantized formats, not pure INT8 and INT4 arithmetic benchmarks.

#### HOWTO use NAX: TensorOps
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

### MLX Metal GEMM: from dispatch to NAX

> Source snapshot: [`3a62199`](https://github.com/ml-explore/mlx/tree/3a6219917e4535575ce5bce2fc2ba27a483a709b). The following blocks are source excerpts, not standalone programs; omitted arguments use `...`.[^mlx-matmul-local]

#### 1. Layer to dispatch

`Linear.__call__` chooses `addmm` when bias is present and `matmul` otherwise:[^mlx-linear]

```python
if "bias" in self:
    x = mx.addmm(self["bias"], x, self["weight"].T)
else:
    x = x @ self["weight"].T
```

In `Matmul::eval_gpu`, layout preparation is followed by shape-based routing:

```cpp
if (M == 1 && N == 1 && batch_size_out == 1 &&
    a.flags().row_contiguous && b.flags().row_contiguous &&
    a.dtype() != complex64) {
  return dot_product(...);
}

if (!a_transposed && b_transposed && gemv_wide(...)) {
  return;
}

if (std::min(M, N) == 1) {
  return gemv(...);
}

return steel_matmul(...);
```

This separates dot products, narrow/wide GEMV cases, and Steel GEMM.

Within Steel, a long reduction dimension `K` can select a split-K kernel. The regular NAX path below is one branch, not the implementation for every matrix multiplication.

#### 2. Select a Metal kernel

NAX eligibility checks both the platform and dtype:

```cpp
bool use_nax = metal::is_nax_available() &&
    !issubdtype(a.dtype(), complexfloating) &&
    (env::enable_tf32() || a.dtype() != float32);
```

`is_nax_available()` checks build support, OS version (26.2 or newer), and GPU architecture. FP16 passes the dtype filter, but still needs a supported platform.[^mlx-device]

For regular NAX GEMM, MLX chooses a threadgroup tile `bm` by `bn`, reduction block `bk`, and SIMD-group arrangement `wm` by `wn`:

```cpp
int bm = 128, bn = 128, bk = 512;
int wm = 4, wn = 4;

char devc = d.get_architecture().back();
if (devc == 's' || devc == 'c' || devc == 'd') {
  bk = (K >= 8192 && K > (M + N)) ? 64 : 256;
  bm = 64;
  wm = 2;
}

kname << "steel_gemm_fused_nax_"
      << (transpose_a ? 't' : 'n')
      << (transpose_b ? 't' : 'n')
      << "_" << type_to_name(a)
      << "_" << type_to_name(out)
      << "_bm" << bm << "_bn" << bn << "_bk" << bk
      << "_wm" << wm << "_wn" << wn;
```

The base kernel name encodes transpose flags, dtypes, and tile sizes. Batch, alignment, and epilogue flags further specialize the cached pipeline.

**Without NAX**, regular Steel GEMM uses `steel_gemm_fused_...`. Its initial tile is `bm = 64, bn = 64, bk = 16`, with `wm = wn = 2`; `GEMM_TPARAM_MACRO` then applies device/shape tuning.[^mlx-non-nax]

#### 3. Map tiles to threadgroups

The output tile counts are rounded up, then swizzled for launch. Shown here is regular NAX GEMM; the non-NAX variant uses `get_steel_gemm_fused_kernel` for pipeline lookup:

```cpp
int tn = (N + bn - 1) / bn;
int tm = (M + bm - 1) / bm;

// GEMMParams retains the logical tile counts and matrix strides.
int tile = 1 << swizzle_log;
tm = (tm + tile - 1) / tile;
tn = tn * tile;

MTL::Size group_dims = MTL::Size(32, wn, wm);
MTL::Size grid_dims = MTL::Size(tn, tm, batch_size_out);

auto& compute_encoder = metal::get_command_encoder(s);
auto kernel = get_steel_gemm_fused_nax_kernel(...);
compute_encoder.set_compute_pipeline_state(kernel);
compute_encoder.set_input_array(a, 0);
compute_encoder.set_input_array(b, 1);
compute_encoder.set_output_array(out, 3);
compute_encoder.set_bytes(params, 4);
// Optional bias/epilogue and batch bindings omitted.
compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
```

`swizzle_log` comes from a device/shape heuristic. Each threadgroup has `32 * wn * wm` threads; `GEMMParams` carries sizes, strides, and the swizzle needed to map the launch back to output tiles. Alignment flags select handling for partial tiles.

The binding indices match the Metal kernel's `[[buffer(0)]]` input A, `[[buffer(1)]]` input B, `[[buffer(3)]]` output D, and `[[buffer(4)]]` parameters. MLX's dispatch wrapper inserts dependency barriers and records the native Metal dispatch:[^mlx-launch]

```cpp
void CommandEncoder::dispatch_threadgroups(
    MTL::Size grid_dims, MTL::Size group_dims) {
  maybeInsertBarrier();
  buffer_ops_++;
  get_command_encoder()->dispatchThreadgroups(grid_dims, group_dims);
}
```

This encodes work into a command buffer. MLX submits it with `buffer_->commit()` when it flushes the stream; dispatch itself does not wait for GPU completion.

**Inside the NAX kernel, `wm/wn` partition the output tile, not the reduction dimension.** The JIT passes host values `bm, bn, bk, wm, wn` as template arguments `BM, BN, BK, WM, WN`. Metal supplies `simd_group_id` via `[[simdgroup_index_in_threadgroup]]`; the kernel uses it to select a sub-tile:[^mlx-nax-partition]

```cpp
// steel_gemm_fused_nax.h: after batch and threadgroup tile offsets.
constexpr short SM = BM / WM;
constexpr short SN = BN / WN;
constexpr short TM = SM / 16;
constexpr short TN = SN / 16;

const short tm = SM * (simd_group_id / WN);
const short tn = SN * (simd_group_id % WN);

A += transpose_a ? tm : (tm * params->lda);
B += transpose_b ? (tn * params->ldb) : tn;
D += tm * params->ldd + tn;

NAXTile<AccumType, TM, TN> Dtile;
// gemm_loop<T, SM, SN, ...> accumulates this tile across K.
```

For `BM = BN = 128` and `WM = WN = 4`, one threadgroup has 16 SIMD-groups (512 threads). Each group computes a `32 x 32` output sub-tile: group 5 starts at `(tm, tn) = (32, 32)` within the threadgroup tile. With `TM = TN = 2`, its accumulator contains a `2 x 2` grid of `16 x 16` fragments. The inner `(16, 32, 16)` operation below is a compute step, not the whole group's tile. Kernel-local `tm/tn` are element offsets, unlike the host's launch tile counts.

#### 4. Load, compute, write

**Non-NAX: shared staging, SIMD-group MMA, global store.** The kernel stages input tiles in threadgroup memory before computing:[^mlx-non-nax]

```cpp
// steel_gemm_fused.h: aligned main loop; offsets and epilogue omitted.
threadgroup T As[gemm_kernel::tgp_mem_size_a];
threadgroup T Bs[gemm_kernel::tgp_mem_size_b];
thread mma_t mma_op(simd_group_id, simd_lane_id);
thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);

for (int k = 0; k < gemm_k_iterations; k++) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  loader_a.load_unsafe();
  loader_b.load_unsafe();
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mma_op.mma(As, Bs);
  loader_a.next();
  loader_b.next();
}
threadgroup_barrier(mem_flags::mem_none);
return mma_op.store_result(D, params->ldd);
```

`mma_op.mma` loads fragments from `As`/`Bs` and reaches this primitive in `mma.h`:

```cpp
METAL_FUNC static constexpr void mma(
    thread mat_type& D, thread mat_type& A,
    thread mat_type& B, thread mat_type& C) {
  simdgroup_multiply_accumulate(D, A, B, C);
}
```

Here `mat_type` is `metal::simdgroup_matrix<T, 8, 8>`: each of 32 threads holds two elements per fragment. This is the non-NAX matrix primitive, distinct from the NAX cooperative-tensor operation below.

**NAX: private tiles, cooperative-tensor MMA, global store.**

Inside `steel/gemm/nax.h`, the inner operation uses a `(16, 32, 16)` descriptor at SIMD-group scope. Cooperative tensors hold operand and accumulator fragments distributed across the group:[^mlx-nax-local][^metal-tensors]

> A fragment is 16x16, so 32 threads will hold kElemsPerFrag = 16x16/32 = 8
>
> However this kernel is 16x32x16 (M, N, K), so it requires 1 A fragment(tile), 2 B fragments, 2 C fragments

```cpp
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    16, 32, 16, transpose_a, transpose_b, true,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

auto ct_a = gemm_op
    .template get_left_input_cooperative_tensor<AType, BType, CType>();
auto ct_b = gemm_op
    .template get_right_input_cooperative_tensor<AType, BType, CType>();
auto ct_c = gemm_op.template get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(ct_a)>,
    metal::remove_addrspace_t<decltype(ct_b)>,
    CType>();

for (short i = 0; i < kElemsPerFrag; i++) {
  ct_a[i] = A[i];
  ct_b[i] = Bn0[i];
  ct_b[kElemsPerFrag + i] = Bn1[i];
  ct_c[i] = Cn0[i];
  ct_c[kElemsPerFrag + i] = Cn1[i];
}

gemm_op.run(ct_a, ct_b, ct_c);

for (short i = 0; i < kElemsPerFrag; i++) {
  Cn0[i] = ct_c[i];
  Cn1[i] = ct_c[kElemsPerFrag + i];
}
```

`run` performs the multiply-accumulate using initialized fragments; the final loop copies the accumulator back to MLX's fragment layout.

**`Cn0[i] = ct_c[i]` copies a private fragment, not an output buffer.** In this helper, `Cn0`/`Cn1` are declared `thread dtype_frag_t<CType>&`; `A` and `Bn0`/`Bn1` are `const thread` references. `dtype_frag_t<U>` is a vector of eight elements per thread.[^mlx-nax-local]

| Storage in the source | Example | Meaning |
| --- | --- | --- |
| `thread` | Inner `A`, `Bn0`, `Cn0` fragments | Per-thread private values, normally register-resident |
| `threadgroup` | Non-NAX `As` / `Bs` | Shared scratchpad for threads in one threadgroup |
| `device` | Outer kernel `A` / `B` / `D` pointers | Global input/output buffers |

The inner fragment named `A` is not the outer kernel's `device` pointer named `A`. `thread` does not guarantee physical register allocation: the compiler can spill. Likewise, a `device` access can hit cache rather than DRAM.[^metal-memory]

The regular NAX path loads from device buffers directly into private tiles, then stores the completed tile through the output device pointer. These excerpts show the missing input/output boundary:[^mlx-nax-memory]

```cpp
// gemm_nax.h: aligned input loads inside the reduction loop.
// A and B are const device T*.
Atile.load(A + A_offset, lda);
Btile.load(B + B_offset, ldb);
tile_matmad_nax(
    Dtile, Atile, metal::bool_constant<transpose_a>{},
    Btile, metal::bool_constant<transpose_b>{});

// steel_gemm_fused_nax.h: after gemm_loop and any epilogue.
// D is device T*; partial output tiles use store_safe instead.
Dtile.store(D, int(params->ldd));
```

So the regular non-NAX path explicitly stages through `threadgroup` memory, while this NAX path does not use that staging step. In both, fragment assignment and global-buffer storage are separate operations.

This is the GPU control surface visible in MLX: dispatch, tiling, launch geometry, and matrix primitives. The Metal compiler still owns instruction lowering.

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
[^llamacpp-m5-max]: ggml-org, [`llama.cpp` Apple Silicon performance discussion](https://github.com/ggml-org/llama.cpp/discussions/4167): M5 Max 40-core legacy Metal reports by [Hassan-A-K](https://github.com/ggml-org/llama.cpp/discussions/4167#discussioncomment-16353087) and [CSendner](https://github.com/ggml-org/llama.cpp/discussions/4167#discussioncomment-16400323), plus the accepted [Tensor API result](https://github.com/ggml-org/llama.cpp/discussions/4167#discussioncomment-18152504). All use Llama 2 7B, `pp512`, `tg128`, and full GPU offload; the legacy pair is averaged in the table.
[^llamacpp-nax-ab]: ggml-org, [`llama.cpp` PR #16634: initial Metal 4 Tensor API support](https://github.com/ggml-org/llama.cpp/pull/16634) and its [M5 branch-versus-main benchmark](https://github.com/ggml-org/llama.cpp/pull/16634#issuecomment-3445563655), October-November 2025. The pull request added Tensor API detection, an opt-out environment variable, and M5-or-later gating.
[^m5-max-ane-measured]: Kaiyu Shi, local M5 Max ANE throughput measurement, September 9, 2026: 25 TFLOPS at FP16 and 50 TOPS at INT8; measured results rather than Apple-published peak specifications.
[^gpu-peak]: Apple, [Apple unleashes M1](https://www.apple.com/newsroom/2020/11/apple-unleashes-m1/), November 2020, reporting 2.6 TFLOPS for M1; Philip Turner, [Apple GPU microarchitecture benchmarks](https://github.com/philipturner/metal-benchmarks), FP32 FLOPS, core-count, and clock tables through Apple8; HubWeb, [A-series](https://hubweb.cn/apple-silicon/chip-a/) and [M-series](https://hubweb.cn/apple-silicon/chip-m/) comparison tables, used for later GPU width and clock inputs. Later values are derived estimates, not Apple-published peak specifications.
[^mlx-linear]: MLX, [`Linear.__call__`](https://github.com/ml-explore/mlx/blob/3a6219917e4535575ce5bce2fc2ba27a483a709b/python/mlx/nn/layers/linear.py#L65-L70).
[^mlx-matmul-local]: MLX, [`Matmul::eval_gpu`](https://github.com/ml-explore/mlx/blob/3a6219917e4535575ce5bce2fc2ba27a483a709b/mlx/backend/metal/matmul.cpp#L1504-L1572), [Steel dispatch](https://github.com/ml-explore/mlx/blob/3a6219917e4535575ce5bce2fc2ba27a483a709b/mlx/backend/metal/matmul.cpp#L911-L1013), and [regular NAX kernel setup](https://github.com/ml-explore/mlx/blob/3a6219917e4535575ce5bce2fc2ba27a483a709b/mlx/backend/metal/matmul.cpp#L205-L317).
[^mlx-device]: MLX, [`is_nax_available()`](https://github.com/ml-explore/mlx/blob/3a6219917e4535575ce5bce2fc2ba27a483a709b/mlx/backend/metal/device.cpp#L947-L966).
[^mlx-nax-local]: MLX, [`nax.h` fragment types](https://github.com/ml-explore/mlx/blob/3a6219917e4535575ce5bce2fc2ba27a483a709b/mlx/backend/metal/kernels/steel/gemm/nax.h#L27-L43) and [cooperative-tensor multiply-accumulate](https://github.com/ml-explore/mlx/blob/3a6219917e4535575ce5bce2fc2ba27a483a709b/mlx/backend/metal/kernels/steel/gemm/nax.h#L386-L455).
[^mlx-non-nax]: MLX, [regular non-NAX dispatch](https://github.com/ml-explore/mlx/blob/3a6219917e4535575ce5bce2fc2ba27a483a709b/mlx/backend/metal/matmul.cpp#L343-L507), [`steel_gemm_fused.h` aligned loop](https://github.com/ml-explore/mlx/blob/3a6219917e4535575ce5bce2fc2ba27a483a709b/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h#L170-L204), and [`mma.h` SIMD-group primitive](https://github.com/ml-explore/mlx/blob/3a6219917e4535575ce5bce2fc2ba27a483a709b/mlx/backend/metal/kernels/steel/gemm/mma.h#L177-L206).
[^mlx-launch]: MLX, [Metal dispatch](https://github.com/ml-explore/mlx/blob/3a6219917e4535575ce5bce2fc2ba27a483a709b/mlx/backend/metal/device.cpp#L408-L414), [command-buffer submission](https://github.com/ml-explore/mlx/blob/3a6219917e4535575ce5bce2fc2ba27a483a709b/mlx/backend/metal/device.cpp#L516-L558), and [stream finalization](https://github.com/ml-explore/mlx/blob/3a6219917e4535575ce5bce2fc2ba27a483a709b/mlx/backend/metal/eval.cpp#L71-L77).
[^mlx-nax-partition]: MLX, [JIT template arguments](https://github.com/ml-explore/mlx/blob/3a6219917e4535575ce5bce2fc2ba27a483a709b/mlx/backend/metal/jit_kernels.cpp#L1084-L1116) and [NAX SIMD-group tile mapping](https://github.com/ml-explore/mlx/blob/3a6219917e4535575ce5bce2fc2ba27a483a709b/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused_nax.h#L150-L202).
[^mlx-nax-memory]: MLX, [`gemm_loop` device loads](https://github.com/ml-explore/mlx/blob/3a6219917e4535575ce5bce2fc2ba27a483a709b/mlx/backend/metal/kernels/steel/gemm/gemm_nax.h#L25-L93) and [NAX output store](https://github.com/ml-explore/mlx/blob/3a6219917e4535575ce5bce2fc2ba27a483a709b/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused_nax.h#L178-L213).
[^metal-memory]: Apple, ["Learn performance best practices for Metal shaders"](https://developer.apple.com/videos/play/tech-talks/111373/), address spaces and caching; [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf), address-space semantics.
[^metal-tensors]: Apple, [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf), tensor operations; see also ["Optimize custom machine learning operations with Metal tensors"](https://developer.apple.com/videos/play/wwdc2026/330/). The linked specification is updated by Apple; MLX excerpts above are pinned to a commit.
[^m-series]: HubWeb, [Apple M-series specification comparison](https://hubweb.cn/apple-silicon/chip-m/): secondary comparison source for M-series generation data.
[^ane-execution]: Apple Neural Engine: A Complete Guide, [Execution model](https://ane-guide.readthedocs.io/en/latest/part-1-machine/02-execution-model.html): reverse-engineered ANE execution context; not an Apple specification.
