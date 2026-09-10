---
layout: post
title: "Apple Silicon AI compute: ANE and GPU"
topic: "AI infrastructure"
sequence: 12
last_modified_at: 2026-09-10T16:04:00+08:00
excerpt: "ANE and GPU hardware, software stacks, and runnable Core ML, PyTorch MPS, MLX, and Metal examples."
description: "A concise hardware and software-stack map of Apple Silicon AI compute across ANE, GPU, Core ML/Core AI, MPS, Metal, MLX, and Metal tensor operations."
tags: "Apple Silicon, AI infrastructure, GPU, ANE, Core ML, Metal, MLX"
---

Apple Silicon has two major AI accelerators: the **Apple Neural Engine (ANE)** and the **GPU**. Their useful distinction is the programming interface: ANE execution is compiler-managed; the GPU also exposes programmable kernels.

## Hardware

<figure class="post-figure">
  <a href="{{ '/assets/apple-soc-block-diagram.svg' | relative_url }}">
    <img src="{{ '/assets/apple-soc-block-diagram.svg' | relative_url }}" alt="Simplified Apple SoC block diagram showing external sensors and interfaces connected through a shared on-chip interconnect to the CPU, GPU, Apple Neural Engine, media engines, memory, and storage." loading="lazy">
  </a>
  <figcaption><strong>Simplified Apple SoC.</strong> Specialized interfaces and compute engines communicate through an on-chip interconnect; this is a functional map, not a physical die floorplan.</figcaption>
</figure>

Both accelerators exist on iPhone and Mac. **The Neural Accelerators inside recent GPU cores are separate from the ANE.**[^a19-pro-spec][^m5-max-spec]

| Hardware | A19 Pro (iPhone 17 Pro) | M5 Max (top configuration) |
| --- | --- | --- |
| GPU cores | 6 | 40 |
| Neural Accelerators in GPU | Yes | Yes |
| ANE cores | 16 | 16 |
| Unified memory bandwidth |  77 GB/s (non-official)   | 614 GB/s |
| GPU FP16 throughput | 12 TFLOPS (peak-guess) | **43 TFLOPS (measured)** / 65 TFLOPS (peak-guess) |
| ANE throughput | FP16: 25 TFLOPS (peak-guess) | **FP16: 25 TFLOPS / INT8: 50 TOPS (measured)** |

The M5 Max ANE numbers are local measurements, not Apple-published peak specifications.[^m5-max-ane-measured]

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
| Public access | CoreML | PyTorch MPS, MLX, Metal; also model frameworks |
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

Tested on an M1 Pro, macOS 26.6, Python with `coremltools 8.3.0`, `torch 2.8.0`, and `mlx 0.32.1`. Outputs below are execution checks, not performance measurements. The linked NAX source walkthrough targets newer GPUs and was not measured on this M1 Pro.

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

## MLX metal_kernel vector add Path

**Write the per-thread computation in Metal; let MLX generate the signature, bind buffers, and launch it.** Here each GPU thread computes one element of `out = a + b`.[^mlx-custom-metal]

```python
import mlx.core as mx

if not mx.metal.is_available():
    raise RuntimeError("A Metal GPU is required")
mx.set_default_device(mx.gpu)

add = mx.fast.metal_kernel(
    name="vector_add",
    input_names=["a", "b"],
    output_names=["out"],
    source="""
        uint i = thread_position_in_grid.x;
        out[i] = a[i] + b[i];
    """,
)
a = mx.arange(256, dtype=mx.float32)
b = mx.full((256,), 10.0, dtype=mx.float32)
(c,) = add(
    inputs=[a, b],
    grid=(a.size, 1, 1),
    threadgroup=(128, 1, 1),
    output_shapes=[a.shape],
    output_dtypes=[a.dtype],
)
mx.eval(c)
max_error = mx.max(mx.abs(c - (a + b))).item()
assert max_error == 0.0
print(mx.device_info()["device_name"])
print(c.shape, c.dtype)
print(c[:8].tolist())
print("max_abs_error:", max_error)
```

Local output (M1 Pro, MLX 0.32.1):

```text
Apple M1 Pro
(256,) mlx.core.float32
[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
max_abs_error: 0.0
```

`source` is Metal code, not Python. `thread_position_in_grid.x` selects indices 0 through 255; `grid` counts **threads**, so this launch has two threadgroups of 128 threads. Each thread loads two FP32 values and writes one; no shared staging or barrier is needed. The grid exactly matches the vector length, so this example needs no bounds guard. This custom kernel bypasses Steel GEMM selection. For the source-level path from `Linear` through Metal dispatch, tiling, and NAX/non-NAX GEMM kernels, see [MLX Metal GEMM: from dispatch to NAX]({% post_url 2026-09-09-apple-gpu-evolution %}#mlx-metal-gemm-from-dispatch-to-nax).

## References

[^mlx-custom-metal]: MLX documentation, [Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html): generated signatures and `dispatchThreads` launch dimensions. Example executed locally with MLX 0.32.1.
[^a19-pro-spec]: Apple Support, ["iPhone 17 Pro and iPhone 17 Pro Max - Technical Specifications"](https://support.apple.com/en-mt/125090).
[^m5-max-spec]: Apple, ["Apple debuts M5 Pro and M5 Max to supercharge the most demanding pro workflows"](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/).
[^m5-max-ane-measured]: Kaiyu Shi, local M5 Max ANE throughput measurement, September 9, 2026: 25 TFLOPS at FP16 and 50 TOPS at INT8; measured results rather than Apple-published peak specifications.
[^ane-paper]: Spencer H. Bryngelson, ["Apple Neural Engine: Architecture, Programming, and Performance"](https://arxiv.org/abs/2606.22283v1). Reverse-engineered findings, not a vendor programming specification.
[^core-ai]: Apple Developer, ["Meet Core AI"](https://developer.apple.com/videos/play/wwdc2026/324/), WWDC26: on-device inference across CPU, GPU, and Neural Engine, with new runtime and export APIs.
[^coreml-compute-units]: Apple coremltools documentation, ["Load and Convert Model Workflow"](https://apple.github.io/coremltools/docs-guides/source/load-and-convert-model.html).
[^coreml-plan]: Apple coremltools, [`MLComputePlan` and `MLComputePlanDeviceUsage`](https://apple.github.io/coremltools/source/coremltools.models.html#module-coremltools.models.compute_plan): anticipated per-operation devices and the preferred compute device. Plan inspection requires macOS 14.4 or newer.
[^pytorch-mps]: PyTorch documentation, ["MPS backend"](https://docs.pytorch.org/docs/stable/notes/mps.html).
