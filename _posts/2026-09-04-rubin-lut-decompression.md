---
layout: post
title: "Rubin LUT decompression"
topic: "GPU architecture"
date: 2026-09-04
excerpt: "Notes on packed 3-bit weights, LUT decompression for MXFP8, quantization granularity, and codebook storage on Rubin."
---

## Key Insights

- Stored in 3bit packed format, do LUT dequantize on the fly, calculated in MXFP8
- Weight only 3-bit, 3.375 bits-per-weight, 20% gains from mxfp4
- **N=8, K=64** forms a block, sharing one codebook (8 entries of FP8 codeword)
- The micro-scaling of MXFP8 is also valid as K=32 per-group scaling

Current Status:

- Hardware: Supported on Rubin
- Software: Developer preview in CUDA13.4[^ptx-lut]

## Comparison with other numeric format

Bits per weights:

| Format | Element / index bits | Scaling bits | Codebook bits | Total bits | Short explanation |
| --- | ---: | ---: | ---: | ---: | --- |
| MXFP8 | 8 | 8/32 = 0.25 | 0 | 8.25 | FP8 values with one E8M0 scale per 32 elements. |
| NVFP4 | 4 | 8/16 = 0.5 | 0 | 4.5 | E2M1 FP4 values with one E4M3 scale per 16 elements; a tensor-level FP32 scale is additional. |
| MXFP4 | 4 | 8/32 = 0.25 | 0 | 4.25 | E2M1 FP4 values with one E8M0 scale per 32 elements. |
| 3bitLUT-MXFP8 | 3 | 8/32 = 0.25 | (8 × 8)/512 = 0.125 | 3.375 | Packed indices select one of eight FP8 codewords shared by 512 weights, with MXFP8 block scaling. |

All numeric columns are amortized bits per weight. The totals follow the accounting above and exclude padding and tensor-level metadata.[^nvfp4][^ptx-lut]

## Benchmark Results

No figures for now

## Tensor Core MMA Overview

The compressed 3bit weight is feed into Tensor Core directly, so memory bandwidth and capacity can be saved on both HBM and SRAM level.

However the peak computation should not increase compared with MXFP8, if the workload already hits compute bound roofline (which is unusual)

<figure class="post-figure">
  <div style="overflow-x: auto;" tabindex="0" aria-label="MMA layout; scroll horizontally on narrow screens">
    <img style="min-width: 640px;" src="{{ '/assets/rubin-lut/mma-overview.png' | relative_url }}" alt="NVIDIA MMA layout showing compressed B indices in shared memory and LUT metadata in Tensor Memory." />
  </div>
  <figcaption>NVIDIA PTX: matrix B usage in computation. Compressed indices and LUT metadata are separate inputs. Scroll horizontally on narrow screens.</figcaption>
</figure>

## Quantize granularity

N=8,K=64(in total 512 elements) shares a single codebook(LUT)[^ptx-lut]

when combined with MXFP8 format, the K=64 can be splitted into 2 scaling groups, each scaled with a FP8-E8M0 scaling factor.

<figure class="post-figure">
  <div style="overflow-x: auto;" tabindex="0" aria-label="Quantization layout; scroll horizontally on narrow screens">
    <img style="min-width: 640px;" src="{{ '/assets/rubin-lut/quantization-block.png' | relative_url }}" alt="NVIDIA diagram showing an N=8, K=64 tile represented by three-bit indices and an eight-entry LUT." />
  </div>
  <figcaption>NVIDIA PTX: one eight-entry LUT for an N=8, K=64 tile. Scroll horizontally on narrow screens.</figcaption>
</figure>

## Codebook storage

CodeBook is stored in 128Byte-aligned format (cache friendly), thus 16 LUT-blocks should maximize the throughput. The 16-blocks is organized as N=64, K=128

<figure class="post-figure">
  <div style="overflow-x: auto;" tabindex="0" aria-label="Codebook layout; scroll horizontally on narrow screens">
    <img style="min-width: 520px;" src="{{ '/assets/rubin-lut/codebook-layout.png' | relative_url }}" alt="NVIDIA GMEM and TMEM codebook layouts with sixteen eight-byte LUTs in each colored group." />
  </div>
  <figcaption>NVIDIA PTX: GMEM and TMEM layout of the lookup table. Each colored group contains sixteen LUTs. Scroll horizontally on narrow screens.</figcaption>
</figure>

## References

[^ptx-lut]: NVIDIA, [PTX ISA 9.4 — Decompression of input matrices](https://docs.nvidia.com/cuda/developer-preview/13.4/parallel-thread-execution/index.html#tcgen05-decompress-inp-mat), CUDA 13.4 developer preview. Source of the three diagrams above.
[^nvfp4]: NVIDIA, [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/): source for the format descriptions in the comparison table.
