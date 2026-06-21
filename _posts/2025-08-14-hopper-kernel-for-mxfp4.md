---
layout: post
title: "Hopper kernel for MXFP4"
topic: "GPU kernels"
sequence: 7
source_url: https://app.notion.com/p/24f2ec4bb1f0806199c1fe74cf9a707b
source_label: "Original note on Notion"
excerpt: "Notes on Triton's Hopper MXFP4 matmul path, from packed FP4 unpacking into BF16 to scale conversion with inline PTX."
---

This note follows the Hopper path for Triton's BF16 x MXFP4 matmul kernel. The entry point is the specialized `matmul_ogs` implementation in Triton, where the usual MoE routing state is combined with MXFP4 weight loading, scale unpacking, and BF16 dot accumulation.

```python
# specialized and simplified triton kernel for bf16 x mxfp4 matmul on Hopper
def matmul_ogs():
    ### MoE Logics
    # ExpertData low-16bit for expert ID (matrix ID),
    # high-16bit for block ID (tiling ID in a single matrix)
    expt_data = tl.load(ExptData + pid_m)
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M = tl.load(ExptHist + expt_id)
    start_m = tl.load(ExptOffs + expt_id)
    start_z = 0
    expt_id, block_id = expt_id.to(index_type), block_id.to(index_type)
    start_m, start_z = start_m.to(index_type), start_z.to(index_type)
    pid_n, pid_k = pid_n.to(index_type), pid_k.to(index_type)

    ### MXFP4 Logics
    x = tl.load(XPtrs, mask=mask_k[None, :], other=0.0)
    w = tl.load(WPtrs, mask=mask_k_w[:, None], other=0.0, cache_modifier=W_CACHE_MODIFIER)
    w_scales = unswizzle_mxfp4_scale_hopper(tl.load(WMxScalePtrs), mx_axis=1, num_warps=num_warps)
    w = mxfp4_to_bf16_triton(w.trans(), w_scales, 1)
    acc = tl.dot(w, x, acc, max_num_imprecise_acc=MAX_NUM_IMPRECISE_ACC, allow_tf32=ALLOW_TF32)
```

The interesting part is that the weight operand is stored compactly as MXFP4, then unpacked and scaled into BF16 before feeding `tl.dot`. The implementation is therefore less about a normal type cast and more about arranging bits so the hardware BF16 instructions can finish the conversion cheaply.

Unpacking FP4 to BF16
---

The path starts from one `uint32` containing eight FP4 values:

```text
1 x uint32 -> 8 x fp4 -> 8 x bf16 -> 4 x bf16x2
```

BF16 is an `E8M7` format, while the FP4 value here is effectively `E2M1`. The all-zero exponent is reserved for subnormal numbers, and the all-ones exponent is reserved for inf and nan. BF16 has bias `127`; FP4 has bias `1`.

That means moving a normal FP4 value into BF16 needs a bias adjustment of `126`, which can be implemented as a BF16 multiply by `2 ** 126`. The mantissa bits keep the same relative position after packing, so the kernel tries to make the unpacking sequence mostly shifts, masks, ors, and BF16 multiply instructions.

The packed layout is deliberately unusual. Four consecutive FP4 values are interleaved so the sign, exponent, and mantissa bits are close to where BF16 wants them:

```python
# -----------------------------------------------------------------------
# Interleave the bits of four consecutive fp4 values (i.e. 16-bits) as:
#     1000000111000000         (first fp4)
#        1000000111000000      (second fp4)
#           1000000111000000   (third fp4)
#     0110110000000000         (fourth fp4)
# This is done so that dequantization can be done in 14 SASS instructions
# -----------------------------------------------------------------------

def _compress_fp4(x):
    x = x.to(torch.int32)
    return ((x & 0x8) << 12) | ((x & 0x7) << 6)


def _compress_fourth(x):
    x = x.to(torch.int32)
    return ((x & 0x8) << 11) | ((x & 0x6) << 9) | ((x & 0x1) << 13)


def _pack_bits(x: torch.Tensor, mx_axis: int):
    x = x.contiguous()
    assert x.shape[-1] % 4 == 0, "Input tensor must have a last dimension divisible by 4"
    x = x.reshape(x.shape[:-1] + (x.shape[-1] // 4, 4))
    first = _compress_fp4(x[..., 0]) | (_compress_fp4(x[..., 0] >> 4) << 16)
    second = _compress_fp4(x[..., 1]) | (_compress_fp4(x[..., 1] >> 4) << 16)
    third = _compress_fp4(x[..., 2]) | (_compress_fp4(x[..., 2] >> 4) << 16)
    fourth = _compress_fourth(x[..., 3]) | (_compress_fourth(x[..., 3] >> 4) << 16)
    x = first | right_shift_unsigned(second, 3) | right_shift_unsigned(third, 6) | fourth
    assert x.is_contiguous()
    x = x.view(torch.uint8)
    return x
```

After packing, a `uint32` can be read as two BF16 lanes at a time:

```text
packed uint32:
  s2,s1,m1,s3,e1,e1,s4,e2 | e2,m2,e3,e3,m3,e4,e4,m4 | ...

for values 2, 3, and 4:

extract mask:
  1,0,0,0,0,0,0,1 | 1,1,0,0,0,0,0,0

result bits:
  s2,0,0,0,0,0,e2,e2 | m2,0,0,0,0,0,0,0
  sign        exponent  mantissa

add bias:
  0x7e80, or 2 ** 126, using mul.bf16x2

for value 1:
  s1 = (x << 1) & 0b1000000000000000
  e1 = (x >> 3) & 0b0000000110000000
  m1 = (x >> 7) & 0b0000000001000000
  then add the BF16 bias with mul.bf16x2
```

On Hopper, `mul.bf16x2` is available directly. The original Triton comment notes that A100 support is still possible by using `fma.bf16x2` with a zero register to emulate the multiply.

```python
def _unpack_fp4_to_bf16_triton(x):
    # For now we implement just H100 support (mul.bf16x2)
    # A100 support is possible via fma
    r0, r1 = tl.inline_asm_elementwise(
        r"""
        {
            .reg .b32 b, c, d<7>, scale;
            .reg .b32 bias;
            mov.b32 bias, 0x7e807e80; // 2 ** 126 == 2 ** (bias_bf16 - bias_fp2)
            // We add the missing bias to the scale directly
            and.b32 $0, $4, 0b10000001110000001000000111000000;
            mul.bf16x2 $0, $0, bias;
            shl.b32 b, $4, 3;
            and.b32 $1, b,  0b10000001110000001000000111000000;
            mul.bf16x2 $1, $1, bias;
            shl.b32 c, $4, 6;
            and.b32 $2, c,  0b10000001110000001000000111000000;
            mul.bf16x2 $2, $2, bias;
            // Unpack last two elements
            shl.b32 d0, $4, 1;
            and.b32 d1, d0, 0b10000000000000001000000000000000;
            shr.b32 d2, $4, 3;
            and.b32 d3, d2, 0b00000001100000000000000110000000;
            or.b32 d4, d1, d3;
            shr.b32 d5, $4, 7;
            and.b32 d6, d5, 0b00000000010000000000000001000000;
            or.b32 $3, d4, d6;
            mul.bf16x2 $3, $3, bias;
        }
        """,
        constraints="=r,=r,=r,=r,r",
        args=[x],
        dtype=(tl.bfloat16, tl.bfloat16),
        is_pure=True,
        pack=4,
    )
    # Concat each pack of 4
    x = tl.join(r0, r1)
    x = x.reshape(x.shape[0], x.shape[1] // 4, 4, x.shape[2])
    x = x.trans(0, 1, 3, 2)
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
    return x
```

Unpacking the scale
---

The MX scale is stored as a `uint8` exponent. The kernel upcasts it to BF16 by using `prmt.b32` to place each byte into the BF16 exponent field, then shifts left by seven bits:

```text
original:
  s1, s2, s3, s4    (u32 = 4 x u8)

zero register:
  0,  0,  0,  0

prmt 0x5140:
  byte select -> 0, s3, 0, s4

shift left by 7:
  align with the BF16 exponent field, so x can be multiplied by scale
  with mul.bf16x2

prmt 0x7362:
  byte select -> 0, s1, 0, s2
```

In Python/Triton form:

```python
# upcast scale to bfloat16
# Add bias missing from the bf16 upcasting sequence
# triton / LLVM generates terrible code for this sequence
# scale = scale.to(tl.uint16)
# scale = scale << 7
# scale = scale.to(tl.bfloat16, bitcast=True)
scale = tl.inline_asm_elementwise(
    r"""
    {
        prmt.b32 $0, $2, 0, 0x5140;
        shl.b32 $0, $0, 7;
        prmt.b32 $1, $2, 0, 0x7362;
        shl.b32 $1, $1, 7;
    }
    """,
    constraints="=r,=r,r",
    args=[scale],
    dtype=tl.bfloat16,
    is_pure=True,
    pack=4,
)
```

The high-level idea is consistent across both paths: make the data layout look almost like BF16 before conversion, then use Hopper's packed BF16 operations to apply the missing exponent bias and scale. That is why the packing looks awkward at first glance; the layout is optimized for the instruction sequence that follows.

References
---

- Triton `matmul_ogs` entry: [python/triton_kernels/triton_kernels/matmul_ogs_details/_matmul_ogs.py](https://github.com/triton-lang/triton/blob/dc5fdc284055ca508c3aa0cdb4da350827689158/python/triton_kernels/triton_kernels/matmul_ogs_details/_matmul_ogs.py#L32)
- Triton Hopper value layout: [python/triton_kernels/triton_kernels/tensor_details/layout_details/hopper_value.py](https://github.com/triton-lang/triton/blob/dc5fdc284055ca508c3aa0cdb4da350827689158/python/triton_kernels/triton_kernels/tensor_details/layout_details/hopper_value.py#L230)
