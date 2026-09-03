---
layout: post
title: "Apple Silicon AI compute: ANE and GPU"
topic: "AI infrastructure"
sequence: 12
last_modified_at: 2026-09-03T16:16:06+08:00
excerpt: "ANE and GPU hardware, runnable Core ML and MLX examples, and the Metal code behind FP16 GEMM."
description: "A concise hardware and software-stack map of Apple Silicon AI compute across ANE, GPU, Core ML/Core AI, MPS, Metal, MLX, and Metal tensor operations."
tags: "Apple Silicon, AI infrastructure, GPU, ANE, Core ML, Metal, MLX"
---

Apple Silicon has two major AI accelerators: the **Apple Neural Engine (ANE)** and the **GPU**. Their useful distinction is the programming interface: ANE execution is compiler-managed; the GPU also exposes programmable kernels.

## Hardware

Both accelerators exist on iPhone and Mac. **The Neural Accelerators inside recent GPU cores are separate from the ANE.**[^a19-pro-spec][^m5-max-spec]

| Hardware | A19 Pro (iPhone 17 Pro) | M5 Max (top configuration) |
| --- | --- | --- |
| GPU cores | 6 | 40 |
| Neural Accelerators in GPU | Yes | Yes |
| ANE cores | 16 | 16 |
| Unified memory bandwidth |  77 GB/s (non-official)   | 614 GB/s |
| GPU / ANE FP16 TFLOPS |  11 TFlops / 19 TFlops (non-official)          | 70 TFlops / 19 TFlops (non-official)    |


<details>
<summary>Die views: A19 Pro and M5 Pro</summary>

<figure class="post-figure">
  <a href="{{ '/assets/uploads/apple-a19-pro-die-shot.png' | relative_url }}"><img src="{{ '/assets/uploads/apple-a19-pro-die-shot.png' | relative_url }}" alt="Annotated A19 Pro die shot showing the single mobile SoC layout."></a>
  <figcaption><strong>A19 Pro.</strong> The annotated die view places CPU, GPU, and ANE on one mobile SoC.</figcaption>
</figure>

<figure class="post-figure">
  <a href="{{ '/assets/uploads/684c52d0-5e90-11f1-8c52-79b37c860eb4.jpg' | relative_url }}"><img src="{{ '/assets/uploads/684c52d0-5e90-11f1-8c52-79b37c860eb4.jpg' | relative_url }}" alt="M5 Pro CPU die annotation with CPU clusters, Neural Engine, and Thunderbolt controllers."></a>
  <figcaption><strong>M5 Pro CPU tile.</strong> The annotation locates the dedicated Neural Engine alongside the CPU clusters.</figcaption>
</figure>

<figure class="post-figure">
  <a href="{{ '/assets/uploads/6ed28cf0-5e90-11f1-8c52-79b37c860eb4.jpg' | relative_url }}"><img src="{{ '/assets/uploads/6ed28cf0-5e90-11f1-8c52-79b37c860eb4.jpg' | relative_url }}" alt="M5 Pro GPU die annotation with GPU cores, Neural Accelerators, LPDDR PHY, display engines, and media engine."></a>
  <figcaption><strong>M5 Pro GPU tile.</strong> The annotations identify GPU cores and their Neural Accelerators. These M5 Pro images illustrate layout, not the M5 Max configuration in the table.</figcaption>
</figure>

</details>

| Dimension | ANE | GPU |
| --- | --- | --- |
| Hardware role | Dedicated neural-network accelerator | Graphics Rendering + Parallel Computation         |
| Typical use | Low-power inference | Inference, training, custom tensor programs |
| Programming unit | Model graph or layer | Tensor op, kernel, layout, command buffer |
| Public access | Compiler placement through model frameworks | PyTorch MPS, MLX, Metal; also model frameworks |
| Custom kernels | No public ANE kernel API | Metal kernels |
| Ops and dtypes | Constrained by compiler and device support | Constrained by framework, Metal, and GPU generation |

For ANE internals beyond the public API, Bryngelson's reverse-engineering study separates measured results from inferred and predicted behavior.[^ane-paper]

## Software Stack

**Model frameworks choose placement; GPU frameworks expose tensor operations and kernels.**

| Entry point | Execution path | Control |
| --- | --- | --- |
| Core ML / Core AI | Compiled model -> CPU / GPU / ANE | Model deployment and hardware placement |
| PyTorch `mps` | PyTorch ops -> MPS / Metal -> GPU | Tensor operations |
| MLX | Array graph -> Metal backend -> GPU (or CPU backend) | Operations and custom kernels |
| Metal | Kernels and command buffers -> GPU | Kernel implementation and dispatch |

Core ML / Core AI represents Apple's model-deployment layer. Core AI evolves that role with updated runtime and export APIs; the example below uses the Core ML `coremltools` API.[^core-ai]

<figure class="post-figure">
  <a href="{{ '/assets/uploads/apple-ml-software-stack.png' | relative_url }}"><img src="{{ '/assets/uploads/apple-ml-software-stack.png' | relative_url }}" alt="Handwritten software stack map from PyTorch and MLX down to MPS, Metal, Tensor Ops, Core ML/Core AI, and Apple accelerators."></a>
  <figcaption><strong>Software stack.</strong> Rough analogies: Core ML/Core AI to QNN-style model deployment; PyTorch's <code>mps</code> device to its <code>cuda</code> device; Metal to CUDA; Metal tensor operations to matrix primitives used by CuTe/Triton. These describe abstraction levels, not API equivalence.</figcaption>
</figure>

Core ML's `CPU_AND_NE` permits CPU and ANE, **excluding GPU**; `CPU_AND_GPU` excludes ANE; `ALL` permits all three. These are allowed targets, not guarantees that an operation will use a particular accelerator.[^coreml-compute-units]

## Minimal Runnable Examples

Tested on an M1 Pro, macOS 26.6, Python with `coremltools 8.3.0`, `torch 2.8.0`, and `mlx 0.32.1`. Outputs below are execution checks, not performance measurements. The newer NAX kernel discussed later is a source walkthrough, not a measured path on this M1 Pro.

### ANE path: PyTorch to Core ML

```python
import coremltools as ct
import numpy as np
import torch


class TinyLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(128, 64)

    def forward(self, x):
        return self.proj(x)


def run_coreml(model, example, package_name):
    traced = torch.jit.trace(model.eval(), example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=example.shape)],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS13,
    )
    mlmodel.save(f"{package_name}.mlpackage")
    print(type(mlmodel).__name__)

    result = mlmodel.predict({"x": example.numpy().astype(np.float32)})
    for name, value in result.items():
        print(name, value.shape, value.dtype)

    # Compute-plan inspection requires macOS 14.4 or newer.
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        mlmodel.get_compiled_model_path(),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    for op in plan.model_structure.program.functions["main"].block.operations:
        if op.operator_name == "const":
            continue
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        device = usage.preferred_compute_device if usage else None
        backend = type(device).__name__ if device is not None else "not reported"
        print(f"{op.operator_name}: planned backend = {backend}")


run_coreml(TinyLinear(), torch.randn(1, 128), "TinyLinear")
```

Local output:

```text
MLModel
var_5 (1, 64) float32
ios16.linear: planned backend = MLCPUComputeDevice
```

For this tiny Linear on the M1 Pro, the plan selects **CPU**, even though ANE is allowed.

Now use a larger convolution module: four `Conv2d + ReLU` stages. Run this after the previous block, reusing `run_coreml`:

```python
layers = []
for i in range(4):
    layers += [
        torch.nn.Conv2d(32 if i == 0 else 64, 64, 3, padding=1),
        torch.nn.ReLU(),
    ]
model = torch.nn.Sequential(*layers)
run_coreml(model, torch.randn(1, 32, 128, 128), "ConvStack")
```

Local output:

```text
MLModel
var_53 (1, 64, 128, 128) float32
ios16.conv: planned backend = MLNeuralEngineComputeDevice
ios16.relu: planned backend = MLNeuralEngineComputeDevice
ios16.conv: planned backend = MLNeuralEngineComputeDevice
ios16.relu: planned backend = MLNeuralEngineComputeDevice
ios16.conv: planned backend = MLNeuralEngineComputeDevice
ios16.relu: planned backend = MLNeuralEngineComputeDevice
ios16.conv: planned backend = MLNeuralEngineComputeDevice
ios16.relu: planned backend = MLNeuralEngineComputeDevice
```

On the same M1 Pro and under the same policy, all eight compute operations now prefer **ANE**.

Backend placement depends on the workload, not just the `compute_units` setting.

### GPU path: PyTorch MPS

```python
import torch

if not torch.backends.mps.is_available():
    raise RuntimeError("PyTorch MPS is not available on this machine")

device = torch.device("mps")
model = torch.nn.Linear(128, 64).to(device)
x = torch.randn(1, 128, device=device)

y = model(x)
torch.mps.synchronize()
print(tuple(y.shape), y.device.type, y.dtype)
```

Local output:

```text
(1, 64) mps torch.float32
```

Here, `.to("mps")` selects the GPU backend directly.[^pytorch-mps]

### GPU path: MLX FP16

```python
import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.gpu)
model = nn.Linear(128, 64)
model.set_dtype(mx.float16)
x = mx.random.normal((1, 128), dtype=mx.float16)

y = model(x)
mx.eval(y)
print(y.shape, y.dtype)
```

Local output:

```text
(1, 64) mlx.core.float16
```

`mx.eval(y)` materializes MLX's lazy computation. This single-row input is a matrix-vector case; the next example uses a full matrix to examine GEMM.

## MLX FP16 Matmul Path

**GEMM becomes a shape-specialized Metal kernel; on supported devices, MLX can use Metal tensor operations for its inner multiply-accumulate.**

Use a bias-free layer to follow `Matmul::eval_gpu` directly. For input `X` of shape `(M, K)` and weight `W` of shape `(N, K)`, it computes `Y = X @ W.T`, an `(M, N)` matrix:

```python
import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.gpu)
model = nn.Linear(128, 64, bias=False)
model.set_dtype(mx.float16)
x = mx.ones((32, 128), dtype=mx.float16)
model.weight = mx.ones((64, 128), dtype=mx.float16)

y = model(x)
mx.eval(y)
print(y.shape, y.dtype)
print(y[0, :4].tolist())
```

Local output:

```text
(32, 64) mlx.core.float16
[128.0, 128.0, 128.0, 128.0]
```

Each output sums 128 products of one. The shape exercises matrix-matrix multiplication; it does not identify the selected kernel.

>  Source snapshot: [`3a62199`](https://github.com/ml-explore/mlx/tree/3a6219917e4535575ce5bce2fc2ba27a483a709b). The following blocks are source excerpts, not standalone programs; omitted arguments use `...`.[^mlx-matmul-local]

### 1. Layer to dispatch

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

### 2. Select a Metal kernel

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

### 3. Map tiles to threadgroups

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

### 4. Load, compute, write

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

## References

[^a19-pro-spec]: Apple Support, ["iPhone 17 Pro and iPhone 17 Pro Max - Technical Specifications"](https://support.apple.com/en-mt/125090).
[^m5-max-spec]: Apple, ["Apple debuts M5 Pro and M5 Max to supercharge the most demanding pro workflows"](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/).
[^ane-paper]: Spencer H. Bryngelson, ["Apple Neural Engine: Architecture, Programming, and Performance"](https://arxiv.org/abs/2606.22283v1). Reverse-engineered findings, not a vendor programming specification.
[^core-ai]: Apple Developer, ["Meet Core AI"](https://developer.apple.com/videos/play/wwdc2026/324/), WWDC26: on-device inference across CPU, GPU, and Neural Engine, with new runtime and export APIs.
[^coreml-compute-units]: Apple coremltools documentation, ["Load and Convert Model Workflow"](https://apple.github.io/coremltools/docs-guides/source/load-and-convert-model.html).
[^coreml-plan]: Apple coremltools, [`MLComputePlan` and `MLComputePlanDeviceUsage`](https://apple.github.io/coremltools/source/coremltools.models.html#module-coremltools.models.compute_plan): anticipated per-operation devices and the preferred compute device. Plan inspection requires macOS 14.4 or newer.
[^pytorch-mps]: PyTorch documentation, ["MPS backend"](https://docs.pytorch.org/docs/stable/notes/mps.html).
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
