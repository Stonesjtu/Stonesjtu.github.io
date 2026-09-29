## Key Insights

- Stored in 3bit packed format, do LUT dequantize on the fly, calculated in MXFP8
- Weight only 3-bit, 3.375 bits-per-weight, 20% gains from mxfp4
- **N=8, K=64** forms a block, sharing one codebook (8 entries of FP8 codeword)
- The micro-scaling of MXFP8 is also valid as K=32 per-group scaling 

Current Status: 

- Hardware: Supported on Rubin
- Software: Developer preview in CUDA13.4

## Comparison with other numeric format

// AGENT: Make this a table, splitting total bits and element bits and scaling bits and codebook bits

// AGENT: add short explaination of each data format in the table

Bits per weights: 

- MXFP8: elements=8 + scaling=8/32 = 8.25 
- NVFP4:  elements=4 + scaling=8/16 = 4.5
- MXFP4: elements=4 + scaling=8/32 = 4.25
- 3bitLUT-MXFP8: elements=3 + scaling=8/32 + codebook=8*8/512 =3.375 

## Benchmark Results

No figures for now

## Tensor Core MMA Overview

The compressed 3bit weight is feed into Tensor Core directly, so memory bandwidth and capacity can be saved on both HBM and SRAM level.

However the peak computation should not increase compared with MXFP8, if the workload already hits compute bound roofline (which is unusual)

![image-20260904104408689](/Users/bytedance/Library/Application Support/typora-user-images/image-20260904104408689.png)



## Quantize granularity

N=8,K=64(in total 512 elements) shares a single codebook(LUT)

when combined with MXFP8 format, the K=64 can be splitted into 2 scaling groups, each scaled with a FP8-E8M0 scaling factor.

![image-20260904104305948](/Users/bytedance/Library/Application Support/typora-user-images/image-20260904104305948.png)

## Codebook storage

CodeBook is stored in 128Byte-aligned format (cache friendly), thus 16 LUT-blocks should maximize the throughput. The 16-blocks is organized as N=64, K=128

![_images/gmem-tmem-layout-lut-b.png](/Users/bytedance/Library/Application Support/typora-user-images/gmem-tmem-layout-lut-b.png)





// polish the style

Reference:

- https://docs.nvidia.com/cuda/developer-preview/13.4/parallel-thread-execution/index.html#tcgen05-decompress-inp-mat
