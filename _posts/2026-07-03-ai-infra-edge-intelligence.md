---
layout: post
title: "How intelligent can the edge become? - AI-Infra Overview PART-4"
topic: "AI infrastructure"
sequence: 11
last_modified_at: 2026-08-09T23:40:14+08:00
source_url: https://app.notion.com/p/38d2ec4bb1f0808ea061d11de43d93a6
source_label: "Original outline on Notion"
excerpt: "A hardware, economics, and model-quality view of how much intelligence can move into phones, AI PCs, and edge AI boxes."
---

This is the fourth part of the series. [Part 1](/2026/06/30/ai-infra-and-tokenomics/) defined infrastructure as the bridge between workload and hardware, [Part 2](/2026/07/01/ai-infra-scaling-problem/) described the inference-demand gap, and [Part 3](/2026/07/02/ai-infra-future/) followed the data-center supply chain.

The edge has a different question: **how much useful intelligence can fit inside one local power, memory, thermal, and price envelope?**

The answer is not a single TOPS number. It is the intersection of four curves:

1. local memory capacity determines which model can load;
2. memory bandwidth limits token generation for weight-streaming workloads;
3. accelerator throughput and compiler coverage determine prefill, perception, and supported operators;
4. small-model quality determines how much capability each stored parameter buys.

## 1. The edge is three different markets

<p class="key-insight"><strong>Key insight</strong><span>Mobile devices, AI PCs, and AI boxes occupy different operating envelopes, so their intelligence must be evaluated against different constraints.</span></p>

"Edge device" hides three distinct products:

| class | typical role | dominant constraints | practical advantage |
| --- | --- | --- | --- |
| **mobile** | private text, speech, vision, and short agent actions | battery, thermal skin temperature, shared DRAM, app lifecycle | always available and closest to personal context |
| **AI PC** | document work, coding, retrieval, media, background assistants | shared-memory capacity and bandwidth, sustained package power | more memory and cooling without a network round trip |
| **AI box** | robotics, cameras, industrial control, local gateways | sensor I/O, deterministic latency, module power, software stack | continuous operation and richer physical-world inputs |

For language models, two first-order bounds are more informative than peak TOPS. If weights use \(b_w\) bytes per parameter, a rough upper bound on resident parameters is:

<div class="math-block">
$$
P_{\max}
\lesssim
\frac{M_{\text{usable}}-M_{\text{runtime}}-M_{\text{KV}}}{b_w}
$$
</div>

During batch-one autoregressive decoding, weights are often streamed from memory for every generated token. A bandwidth ceiling is therefore:

<div class="math-block">
$$
\text{tokens/s}
\lesssim
\frac{B_{\text{memory}}}
{P_{\text{active}}b_w+\text{KV bytes per token}}
$$
</div>

Here \(M\) is memory capacity, \(B_{\text{memory}}\) is sustained memory bandwidth, and \(P_{\text{active}}\) is the parameter count touched per token. These are ceilings, not forecasts: the OS, runtime, activations, KV cache, unsupported operators, and thermal throttling all reduce what a product sustains. Apple has independently shown the same bandwidth-bound behavior for short-sequence Transformers on its Neural Engine.[^apple-ane]

## 2. Hardware evolved along different metric contracts

<p class="key-insight"><strong>Key insight</strong><span>Memory capacity, memory bandwidth, and neural throughput improved across every edge class, but vendor metrics are only comparable within a shared precision and sparsity contract.</span></p>

<figure class="post-figure post-chart">
  <div class="chart-grid">
    <section class="chart-panel">
      <p class="chart-title">AI-box memory capacity</p>
      <p class="chart-subtitle">NVIDIA history plus 2026 Chinese-system snapshot, GB</p>
      <div class="chart-frame"><canvas id="edge-box-memory-chart" role="img" aria-label="Chart comparing memory capacity across NVIDIA AI boxes from Jetson TK1 through DGX Spark and Jetson Thor, with current Huawei and Orange Pi systems.">The representative data is available in the table below.</canvas></div>
    </section>
    <section class="chart-panel">
      <p class="chart-title">AI-box memory bandwidth</p>
      <p class="chart-subtitle">Vendor-disclosed peak GB/s</p>
      <div class="chart-frame"><canvas id="edge-box-bandwidth-chart" role="img" aria-label="Chart showing disclosed memory bandwidth for NVIDIA Jetson systems, DGX Spark, and Huawei Atlas 200I DK A2.">The representative data is available in the table below.</canvas></div>
    </section>
    <section class="chart-panel">
      <p class="chart-title">AI-PC memory capacity</p>
      <p class="chart-subtitle">Maximum addressable unified or system memory, GB</p>
      <div class="chart-frame"><canvas id="edge-pc-memory-chart" role="img" aria-label="Chart comparing maximum memory capacity for MacBook Pro M1 through M5 Pro and Max, Snapdragon X Elite, and AMD Ryzen AI Max systems.">The representative data is available in the table below.</canvas></div>
    </section>
    <section class="chart-panel">
      <p class="chart-title">AI-PC memory bandwidth</p>
      <p class="chart-subtitle">Apple, AMD, and Qualcomm platform peaks, GB/s</p>
      <div class="chart-frame"><canvas id="edge-pc-bandwidth-chart" role="img" aria-label="Chart comparing memory bandwidth for MacBook Pro M1 through M5 Pro and Max, Snapdragon X Elite, and AMD Ryzen AI Max systems.">The representative data is available in the table below.</canvas></div>
    </section>
    <section class="chart-panel">
      <p class="chart-title">Mobile SoC AI evolution</p>
      <p class="chart-subtitle">Apple and Qualcomm on one 2017-2026 timeline; logarithmic within-vendor indices</p>
      <div class="chart-frame chart-frame--mobile-soc"><canvas id="edge-mobile-soc-chart" role="img" aria-label="Unified logarithmic chart placing Apple A11 through A19 Pro and Qualcomm Snapdragon 855 through Snapdragon 8 Gen 5 and 8 Elite Gen 5 on one timeline. Estimated 15 percent bridge factors connect A15 to A16 and Snapdragon 888 to 8 Gen 2.">The source metrics, bridge estimates, and sensitivity ranges are available below.</canvas></div>
    </section>
  </div>
  <figcaption>Capacity and bandwidth are the most portable measurements across these families. The unified mobile plot shares one time axis, but Apple and Qualcomm retain separate baselines: A11 = 1 for Apple and Snapdragon 855 = 1 for Qualcomm, so cross-vendor vertical positions are not comparable. Solid lines use the older disclosed contracts. Dashed colored lines bridge the contract changes by estimating A16 at 15% above A15 and Snapdragon 8 Gen 2 at 15% above Snapdragon 888, then propagate the later Geekbench ratios or Qualcomm NPU claims. A 10-20% bridge assumption would place A19 Pro at roughly 63.6-69.4x its A11 baseline and Elite Gen 5 at roughly 16.1-17.5x its Snapdragon 855 baseline; the plotted midpoints are 66.5x and 16.8x. The orange diamond is the separate 8 Gen 5 branch. Gray dashed extensions mark products still current in August 2026, not new performance samples. Recent Snapdragon points are slightly shifted within each year for readability. Chinese AI boxes are placed at 2026 as a current-catalog snapshot. Blank bandwidth fields remain unplotted.</figcaption>
</figure>

### AI boxes became local generative computers

Jetson TK1 began in 2014 as a 2 GB CUDA development board with 326 GFLOP/s FP32. Xavier NX reached 8 GB and 51.2 GB/s in 2019; Orin Nano Super doubled that bandwidth to 102 GB/s. The 2025 desktop-class systems then jumped to a 128 GB coherent memory pool: DGX Spark couples 273 GB/s with up to 1 sparse FP4 PFLOP, while Jetson AGX Thor pairs the same capacity and bandwidth with the sensor I/O and 40-130 W envelope required by physical AI.[^nvidia-ai-box-family]

Chinese systems now span a similarly wide product range. Huawei's Atlas 200I DK A2 exposes 12 GB and 51.2 GB/s with 20 INT8 TOPS; Orange Pi AIpro packages the same Ascend-oriented edge software ecosystem with up to 24 GB and 20 advertised TOPS; Orange Pi AI Station reaches 96 GB and 176 advertised TOPS in a compact local-inference system.[^china-ai-boxes] These are useful capacity points, but they are not one clean compute curve: model, sparsity, precision, and even whether TOPS covers one accelerator or a complete SoC differ.

The compute labels changed because the hardware changed. FP32, sparse INT8, and sparse FP4 should not be placed on one arithmetic speedup line. The comparable structural trend is broader: **more local state, more bandwidth, narrower numerical formats, and much more dedicated tensor hardware.**

### AI PCs made shared memory part of the accelerator

MacBook Pro shows why memory is the better local-LLM metric. The Pro line moved from 32 GB at 200 GB/s on M1 Pro to 64 GB at 307 GB/s on M5 Pro. The Max line grew from 64 GB at 400 GB/s on M1 Max to 128 GB at 614 GB/s on M5 Max. Between them, M3 Pro briefly traded bandwidth for efficiency, while M4 Pro/Max and M5 Pro/Max resumed scaling both capacity and bandwidth.[^macbook-pro-series]

The base M5 MacBook Pro adds a different inflection: 32 GB at 153 GB/s, with a Neural Accelerator in every GPU core rather than relying only on the 16-core Neural Engine. Apple reports more than 4x M4's peak GPU AI compute, but does not publish that result as a directly comparable TOPS figure.[^m5] On the Windows side, Snapdragon X Elite supports up to 64 GB at 136 GB/s with a 45-TOPS NPU, while AMD Ryzen AI Max systems support up to 128 GB at 256 GB/s, up to 96 GB assignable as graphics memory, and a 50-TOPS NPU.[^snapdragon-x][^amd-ai-max][^amd-halo]

For local LLMs, the important AI-PC feature is not an isolated NPU score. It is a heterogeneous package in which CPU, GPU, NPU, and media engines can share a relatively large memory pool. An NPU may excel at a compiled subgraph while the GPU or CPU handles unsupported operators and sampling.

### Mobile delivered the cleanest same-precision curve

Apple disclosed FP16 Neural Engine throughput from A11 through A15: 0.6, 5.4, 5.4, 11.66, and 15.8 TFLOP/s from 2017 to 2021, a 26x increase in four years.[^apple-ane] A16 then moved Apple's public wording to nearly 17 trillion operations per second; A17 Pro was described as up to 2x faster, A18 as running ML models up to 2x faster than A16, and A19 Pro added Neural Accelerators to every GPU core without an absolute Neural Engine rate.[^apple-a16-a19] Because those claims no longer form one arithmetic contract, the chart estimates A16 at 1.15x A15, then applies the relative Geekbench AI Core ML Neural Engine quantized scores for A16 through A19 Pro: 22,870, 31,040, 45,083, and 50,200.[^apple-geekbench-ai] This produces a continuous directional curve without pretending the bridge is measured. As of August 4, 2026, A19 and A19 Pro remain Apple's current announced A-series generation.

Qualcomm published a separate same-vendor aggregate series: Snapdragon 855, 865, and 888 rose from 7 to 15 to 26 total AI Engine TOPS between 2018 and 2020.[^snapdragon-ai-engine] For the recent flagship sequence, Qualcomm stopped publishing that same aggregate TOPS measure. The chart estimates Snapdragon 8 Gen 2 at 1.15x Snapdragon 888, then compounds the disclosed Hexagon NPU gains: 98% for 8 Gen 3, 45% for Snapdragon 8 Elite, and 37% for 8 Elite Gen 5.[^snapdragon-8-series] Snapdragon 8 Elite is the released product that Qualcomm says would otherwise have been called "8 Gen 4." Qualcomm's current catalog also includes Snapdragon 8 Gen 5 as a separate branch: its 46% NPU uplift over 8 Gen 3 is applied to the same estimated bridge.[^snapdragon-8-gen5] This claim chain is useful for reading cadence, but it is not a controlled benchmark because each percentage can use a different workload mix.

Deployment capability moved with the curves. Snapdragon 8 Gen 3 supports on-device generative models up to 10 billion parameters; Snapdragon 8 Elite Gen 5 advertises up to 20 tokens/s.[^snapdragon-8g3][^snapdragon-elite-gen5] MediaTek's Dimensity 9400 reported 80% faster LLM prompt processing and 35% better efficiency than Dimensity 9300; Dimensity 9500 then doubled NPU compute and added native 1.58-bit BitNet processing.[^mediatek-9400][^mediatek-9500] These are same-vendor generation claims, not cross-vendor benchmark scores.

<details class="post-details" markdown="1">
<summary>Show representative hardware data</summary>

| class | product | year | memory capacity | memory bandwidth | disclosed neural compute | launch price basis |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AI box | Jetson TK1 dev kit | 2014 | 2 GB | 14.9 GB/s | 0.326 FP32 TFLOP/s | USD 192 |
| AI box | Jetson Xavier NX | 2019 | 8 GB | 51.2 GB/s | 21 sparse INT8 TOPS | USD 399 |
| AI box | Jetson Orin Nano Super | 2024 | 8 GB | 102 GB/s | 67 sparse INT8 TOPS | USD 249 |
| AI box | NVIDIA DGX Spark | 2025 | 128 GB | 273 GB/s | 1,000 sparse FP4 TFLOP/s | USD 4,699 current store price |
| AI box | Jetson AGX Thor dev kit | 2025 | 128 GB | 273 GB/s | 2,070 sparse FP4 TFLOP/s | USD 3,499 |
| AI box | Huawei Atlas 200I DK A2 | current catalog | up to 12 GB | up to 51.2 GB/s | 20 INT8 TOPS / 10 FP16 TFLOP/s | module / kit dependent |
| AI box | Orange Pi AIpro 20T | current catalog | up to 24 GB | not disclosed | 20 advertised TOPS | channel price |
| AI box | Orange Pi AI Station | current catalog | up to 96 GB | not disclosed | 176 advertised TOPS | channel price |
| AI PC | MacBook Pro M1 Pro / Max | 2021 | 32 / 64 GB | 200 / 400 GB/s | 16-core Neural Engine | product configuration |
| AI PC | MacBook Pro M2 Pro / Max | 2023 | 32 / 96 GB | 200 / 400 GB/s | 16-core Neural Engine | product configuration |
| AI PC | MacBook Pro M3 Pro / Max | 2023 | 36 / 128 GB | 150 / 400 GB/s | 16-core Neural Engine | product configuration |
| AI PC | MacBook Pro M4 Pro / Max | 2024 | 64 / 128 GB | 273 / 546 GB/s | 16-core Neural Engine | product configuration |
| AI PC | Snapdragon X Elite platform | 2024 | up to 64 GB | 136 GB/s | 45 NPU TOPS | device-dependent |
| AI PC | AMD Ryzen AI Max 385 / 390 / Max+ 395 | 2025 | up to 128 GB | up to 256 GB/s | 50 NPU TOPS; 32 / 32 / 40 GPU CUs | device-dependent |
| AI PC | MacBook Pro M5 | 2025 | up to 32 GB | 153 GB/s | 16-core Neural Engine plus GPU Neural Accelerators | starts at USD 1,699 in 2026 |
| AI PC | MacBook Pro M5 Pro / Max | 2026 | 64 / 128 GB | 307 / 614 GB/s | 16-core Neural Engine plus GPU Neural Accelerators | starts at USD 2,199 / 3,599 |
| mobile | iPhone X / A11 | 2017 | 3 GB device DRAM | not disclosed | 0.6 FP16 TFLOP/s | USD 999 |
| mobile | iPhone XS / A12 | 2018 | 4 GB device DRAM | not disclosed | 5.4 FP16 TFLOP/s | USD 999 |
| mobile | iPhone 11 Pro / A13 | 2019 | 4 GB device DRAM | not disclosed | 5.4 FP16 TFLOP/s | USD 999 |
| mobile | iPhone 12 Pro / A14 | 2020 | 6 GB device DRAM | not disclosed | 11.66 FP16 TFLOP/s | USD 999 |
| mobile | iPhone 13 Pro / A15 | 2021 | 6 GB device DRAM | not disclosed | 15.8 FP16 TFLOP/s | USD 999 |
| mobile | Snapdragon 855 / 865 / 888 | 2018-2020 | device-dependent | device-dependent | 7 / 15 / 26 aggregate AI Engine TOPS | device-dependent |
| mobile | iPhone 14 Pro / A16 Bionic | 2022 | 6 GB device DRAM | not disclosed | nearly 17 trillion operations/s; Geekbench AI Q 22,870 | USD 999 |
| mobile | iPhone 15 Pro / A17 Pro | 2023 | 8 GB device DRAM | not disclosed | up to 2x faster Neural Engine; Geekbench AI Q 31,040 | USD 999 |
| mobile | Snapdragon 8 Gen 3 | 2023 | device-dependent | LPDDR5X platform | up to 10B on-device model | device-dependent |
| mobile | iPhone 16 Pro / A18 Pro | 2024 | 8 GB device DRAM | not disclosed | up to 2x A16 ML speed; Geekbench AI Q 45,083 | USD 999 |
| mobile | Snapdragon 8 Elite (the released Gen 4) | 2024 | device-dependent | LPDDR5X platform | 45% faster Hexagon NPU | device-dependent |
| mobile | iPhone 17 Pro / A19 Pro | 2025; current Aug. 2026 | 12 GB device DRAM | not disclosed | 16-core Neural Engine plus per-GPU-core accelerators; Geekbench AI Q 50,200 | USD 1,099 |
| mobile | Snapdragon 8 Gen 5 | 2025; current Aug. 2026 catalog | up to 24 GB | LPDDR5X up to 4,800 MHz | 46% faster NPU than 8 Gen 3 | device-dependent |
| mobile | Snapdragon 8 Elite Gen 5 | 2025; current Aug. 2026 flagship | device-dependent | LPDDR5X platform | up to 20 LLM tokens/s; 37% faster NPU | device-dependent |
| mobile | Dimensity 9400 / 9500 | 2024-2025 | device-dependent | LPDDR5X platform | +80% prompt speed, then 2x NPU compute | device-dependent |

Mobile DRAM capacities are device-level teardown/specification values rather than Apple disclosures; they are included only to show the resident-memory envelope.[^iphone-memory]

</details>

## 3. Price per performance fell, but only inside each class

<p class="key-insight"><strong>Key insight</strong><span>Edge intelligence became cheaper, but dollars per performance remain meaningful only within the same device class and compute contract.</span></p>

<figure class="post-figure post-chart">
  <div class="chart-grid">
    <section class="chart-panel">
      <p class="chart-title">AI box</p>
      <p class="chart-subtitle">USD per sparse INT8 TOPS</p>
      <div class="chart-frame chart-frame--compact"><canvas id="edge-box-economics-chart" role="img" aria-label="Logarithmic line chart showing AI-box launch dollars per sparse INT8 TOPS falling from 19 dollars for Xavier NX to 3.72 dollars for Orin Nano Super.">The calculations are available in the table below.</canvas></div>
    </section>
    <section class="chart-panel">
      <p class="chart-title">AI PC</p>
      <p class="chart-subtitle">Whole-laptop USD per advertised Apple NPU TOPS</p>
      <div class="chart-frame chart-frame--compact"><canvas id="edge-pc-economics-chart" role="img" aria-label="Logarithmic line chart showing MacBook Air dollars per advertised NPU TOPS falling from about 91 dollars on M1 to about 26 dollars on M4.">The calculations are available in the table below.</canvas></div>
    </section>
    <section class="chart-panel">
      <p class="chart-title">Mobile</p>
      <p class="chart-subtitle">Whole-device USD per FP16 Neural Engine TFLOP/s</p>
      <div class="chart-frame chart-frame--compact"><canvas id="edge-mobile-economics-chart" role="img" aria-label="Logarithmic line chart showing iPhone Pro launch dollars per FP16 Neural Engine TFLOP per second falling from 1665 dollars on A11 to about 63 dollars on A15.">The calculations are available in the table below.</canvas></div>
    </section>
  </div>
  <figcaption>Product-level price proxies at launch. The panels deliberately use different contracts: USD per sparse INT8 TOPS for comparable edge modules, USD per advertised Apple NPU TOPS for MacBook Air, and whole-device USD per FP16 TFLOP/s for the A11-A15 iPhone Pro series. These curves show what a buyer received, not the manufacturing cost of the accelerator alone.</figcaption>
</figure>

The expanded product families are intentionally absent from these three cost curves unless their denominator matches an existing contract. DGX Spark and Jetson Thor publish sparse FP4 throughput; Orange Pi and Huawei publish INT8 or aggregate AI TOPS; AMD AI Max and Snapdragon X emphasize NPU TOPS even though local LLM decoding often runs on the GPU. MacBook Pro M5 shifts still more AI compute into per-GPU-core Neural Accelerators without publishing an absolute TOPS value. Plotting all of them as one USD/TOPS series would reward whichever vendor chose the narrowest precision and most generous sparsity convention.

Three trends survive the imperfect pricing data:

1. Comparable dedicated INT8 edge modules fell from roughly USD 19 per advertised TOPS for Xavier NX to USD 3.72 for Orin Nano Super.
2. Apple kept the base MacBook Air at USD 999 while advertised Neural Engine throughput rose from 11 TOPS on M1 to 38 TOPS on M4, reducing the whole-laptop proxy from about USD 91 to USD 26 per NPU TOPS.
3. In Apple's disclosed FP16 mobile series, the iPhone Pro purchase proxy fell from about USD 1,665 per FP16 TFLOP/s for A11 to about USD 63 for A15.

The denominator is still peak performance. Real model throughput can be lower because operator coverage, quantization format, memory bandwidth, and thermals differ. Phone SoC BOM estimates are even less stable: Apple does not sell A-series chips, and Qualcomm prices are negotiated by volume. Product launch price is therefore the more reproducible historical anchor, despite bundling displays, batteries, storage, and many other components.

<details class="post-details" markdown="1">
<summary>Show price/performance calculations</summary>

| class | product | price | comparable compute | product-level USD per performance |
| --- | --- | ---: | ---: | ---: |
| AI box | Xavier NX | USD 399 | 21 sparse INT8 TOPS | USD 19.00 / TOPS |
| AI box | Orin Nano Super | USD 249 | 67 sparse INT8 TOPS | USD 3.72 / TOPS |
| AI PC | M1 MacBook Air | USD 999 | 11 advertised NPU TOPS | USD 90.82 / TOPS |
| AI PC | M4 MacBook Air | USD 999 | 38 advertised NPU TOPS | USD 26.29 / TOPS |
| mobile | iPhone X / A11 | USD 999 | 0.6 FP16 TFLOP/s | USD 1,665 / TFLOP/s |
| mobile | iPhone XS / A12 | USD 999 | 5.4 FP16 TFLOP/s | USD 185 / TFLOP/s |
| mobile | iPhone 11 Pro / A13 | USD 999 | 5.4 FP16 TFLOP/s | USD 185 / TFLOP/s |
| mobile | iPhone 12 Pro / A14 | USD 999 | 11.66 FP16 TFLOP/s | USD 85.68 / TFLOP/s |
| mobile | iPhone 13 Pro / A15 | USD 999 | 15.8 FP16 TFLOP/s | USD 63.23 / TFLOP/s |

The iPhone Pro line is used to hold launch price at USD 999 across the FP16 series. This is a whole-device affordability proxy, not SoC cost.[^iphone-x][^iphone-xs][^iphone-11-pro][^iphone-12-pro][^iphone-13-pro]

</details>

## 4. Small models are buying more intelligence per parameter

<p class="key-insight"><strong>Key insight</strong><span>Small-model quality and quantization are improving the intelligence that fits inside a fixed local memory budget.</span></p>

<figure class="post-figure post-chart">
  <div class="chart-grid chart-grid--two">
    <section class="chart-panel">
      <p class="chart-title">Small-model MMLU</p>
      <p class="chart-subtitle">Vendor-reported, five-shot</p>
      <div class="chart-frame"><canvas id="edge-model-mmlu-chart" role="img" aria-label="Line chart showing vendor-reported five-shot MMLU increasing from 56.7 for Phi-2 in 2023 to 67.3 for Phi-4 Mini in 2025.">The benchmark data is available in the table below.</canvas></div>
    </section>
    <section class="chart-panel">
      <p class="chart-title">Llama 3.2 3B quantization retention</p>
      <p class="chart-subtitle">QLoRA score divided by BF16 score</p>
      <div class="chart-frame"><canvas id="edge-model-retention-chart" role="img" aria-label="Horizontal bar chart showing quantized Llama 3.2 3B retaining about 98 to 100 percent of BF16 scores on MMLU, GSM8K, and IFEval.">The benchmark data is available in the table below.</canvas></div>
    </section>
  </div>
  <figcaption>Left: directional five-shot MMLU progress for 2.7-3.8B models using each vendor's published evaluation. Right: Meta's same-model, same-harness comparison shows that optimized quantization retained 98-101% of BF16 scores across MMLU, GSM8K, and instruction following. Cross-vendor points are evidence of a capability trend, not a leaderboard.</figcaption>
</figure>

Hardware is only half the edge curve. The other half is model quality at a size that can actually fit.

In late 2023, Microsoft's 2.7B Phi-2 reported 56.7 on five-shot MMLU. Meta's 2024 Llama 3.2 3B BF16 instruction model reported 63.4, and Microsoft's 2025 Phi-4 Mini 3.8B reported 67.3.[^phi2][^llama32][^phi4mini] Evaluation harnesses and training data differ, so the exact slope is uncertain. The broad result is still useful: a roughly 3-4B parameter budget crossed from compact completion model into instruction following, tool use, and materially stronger reasoning.

Quantization makes that progress deployable. Meta's Llama 3.2 3B results show MMLU moving from 63.4 in BF16 to 62.4 with QLoRA quantization, GSM8K from 77.7 to 77.9, and IFEval from 77.4 to 75.9. Its optimized quantized release reduced model size by 56% and runtime memory by 41% on average while improving decode latency by 2.5x.[^llama32-quant]

<details class="post-details" markdown="1">
<summary>Show edge-fit model benchmark data</summary>

| model | release | parameters | five-shot MMLU | GSM8K | instruction / tool signal | approximate Q4 weight bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phi-2 | 2023 | 2.7B | 56.7 | not separately reported in source | research base model | 1.35 GB |
| Llama 3.2 3B Instruct BF16 | 2024 | 3.2B | 63.4 | 77.7 | IFEval 77.4; BFCL v2 67.0 | 1.6 GB |
| Llama 3.2 3B Instruct QLoRA | 2024 | 3.2B | 62.4 | 77.9 | IFEval 75.9; BFCL v2 63.5 | 1.6 GB plus scales/metadata |
| Phi-4 Mini Instruct | 2025 | 3.8B | 67.3 | 88.6 | Arena Hard 32.8 | 1.9 GB |

Scores come from vendor model cards and are not fully standardized across rows. Approximate Q4 bytes use 0.5 byte per parameter before quantization metadata, embedding/output exceptions, runtime buffers, and KV cache.

</details>

Benchmark quality is also becoming more task-specific. A local model does not need to memorize the entire web to be valuable. It can use private local context, retrieval, deterministic tools, and a cloud escalation path. The important edge benchmark is eventually not MMLU alone; it is successful local task completion under a latency, energy, memory, and privacy budget.

## 5. Prediction: the edge becomes a hierarchy of agents

<p class="key-insight"><strong>Key insight</strong><span>Edge devices will become the private, low-latency first tier of hierarchical agents, escalating only work that benefits from cloud-scale models and context.</span></p>

The hardware and model curves compound. More memory and bandwidth allow a larger or less aggressively quantized model; better training and distillation increase quality at the same size; quantization returns capacity and speed to the system.

A practical 2026 envelope looks like this:

| device envelope | plausible local model class | intelligence that fits well | likely cloud boundary |
| --- | --- | --- | --- |
| **8-12 GB mobile** | 1-4B Q4, with short or managed context | private summarization, rewriting, classification, speech/vision understanding, tool routing | broad knowledge, long reasoning, large multimodal context |
| **16-32 GB AI PC** | 7-14B Q4, or several smaller specialists | document RAG, coding assistance, personal search, richer tool use | frontier-quality generation and long-running multi-agent work |
| **64-128 GB AI PC / AI box** | 30-70B Q4 or sparse multimodal models | local enterprise knowledge, robotics planning, multi-camera perception, resilient offline agents | rare high-complexity planning and shared global state |

These ranges reserve memory for the operating system, runtime, activations, and KV cache; they are placement envelopes rather than product guarantees.

<figure class="post-figure">
  <img src="{{ '/assets/edge-cloud-partition-excalidraw.svg' | relative_url }}" alt="Excalidraw placement map showing an AI runtime routing work between private, latency-sensitive edge inference and cloud systems with larger models, long context, tools, and shared state.">
  <figcaption>The edge/cloud boundary becomes hierarchical: keep identity, sensors, private state, and fast actions local; escalate only the work that benefits from larger models, long context, shared knowledge, or elastic compute.</figcaption>
</figure>

My prediction is that edge devices will not become miniature copies of the cloud. They will become the **first tier of a hierarchical agent system**:

- mobile owns identity, private context, sensors, and instant actions;
- the AI PC owns the user's larger working set and sustained local workflows;
- the AI box owns continuous physical-world perception and control;
- the cloud supplies frontier reasoning, broad shared knowledge, and elastic compute.

The most important edge-infra work will therefore sit between model and device: quantization, speculative decoding, memory planning, KV-cache policy, heterogeneous operator placement, thermal-aware scheduling, retrieval, and cloud routing. Peak TOPS will continue to rise, but **the intelligence users feel will be determined by how efficiently the system converts local bytes, watts, and benchmark quality into completed work.**

## References

[^apple-ane]: Apple Machine Learning Research, [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers), 2022.
[^apple-a16-a19]: Apple, [iPhone 14 Pro with A16 Bionic](https://www.apple.com/ie/newsroom/2022/09/apple-debuts-iphone-14-pro-and-iphone-14-pro-max/), 2022; [iPhone 15 Pro with A17 Pro](https://www.apple.com/newsroom/2023/09/apple-unveils-iphone-15-pro-and-iphone-15-pro-max/), 2023; [iPhone 16 with A18](https://www.apple.com/newsroom/2024/09/apple-introduces-iphone-16-and-iphone-16-plus/), 2024; [iPhone 17 Pro with A19 Pro](https://www.apple.com/newsroom/2025/09/apple-unveils-iphone-17-pro-and-iphone-17-pro-max/), 2025.
[^apple-geekbench-ai]: Primate Labs, [Geekbench AI benchmark chart](https://browser.geekbench.com/ai-benchmarks), accessed 2026-08-04. The article uses representative aggregate Core ML Neural Engine quantized scores for iPhone 14 Pro A16 Bionic, iPhone 15 Pro A17 Pro, iPhone 16 Pro A18 Pro, and iPhone 17 Pro Max A19 Pro; this is a device benchmark, not peak hardware throughput.
[^nvidia-ai-box-family]: NVIDIA, [Jetson TK1 Platform Brief](https://developer.download.nvidia.com/embedded/jetson/TK1/docs/Jetson_platform_brief_May2014.pdf), 2014; [Jetson Xavier NX launch specifications](https://nvidianews.nvidia.com/_gallery/download_pdf/5dc30ac0ed6ae5472caf232e/), 2019; [Jetson Orin Nano Super Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/), accessed 2026-07-30; [DGX Spark specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/), accessed 2026-08-04; [DGX Spark current U.S. store listing](https://marketplace.nvidia.com/en-us/enterprise/personal-ai-supercomputers/dgx-spark/), accessed 2026-08-04; [Introducing NVIDIA Jetson Thor](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/), 2025.
[^china-ai-boxes]: Huawei, [Atlas 200I A2 accelerator module specifications](https://e.huawei.com/cn/products/computing/ascend/atlas-200-ai), accessed 2026-08-04; Orange Pi, [Orange Pi AIpro 20T specifications](https://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-AIpro%2820t%29.html) and [Orange Pi AI Station specifications](https://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-AI-Station.html), accessed 2026-08-04.
[^macbook-pro-series]: Apple, [M1 Pro and M1 Max](https://www.apple.com/uk/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/), 2021; [M2 Pro and M2 Max](https://www.apple.com/newsroom/2023/01/apple-unveils-macbook-pro-featuring-m2-pro-and-m2-max/), 2023; Apple Support, [M3 Pro and M3 Max technical specifications](https://support.apple.com/en-la/117736), 2023; Apple, [M4 Pro and M4 Max](https://www.apple.com/ca/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/), 2024; [M5 Pro and M5 Max](https://www.apple.com/newsroom/2026/03/apple-introduces-macbook-pro-with-all-new-m5-pro-and-m5-max/), 2026.
[^m5]: Apple, [Apple unleashes M5](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/), 2025.
[^snapdragon-x]: Qualcomm, [Snapdragon X Elite specifications](https://www.qualcomm.com/laptops/products/snapdragon-x-elite), accessed 2026-07-30.
[^amd-ai-max]: AMD, [Ryzen AI Max and Ryzen AI Max PRO announcement](https://www.amd.com/en/newsroom/press-releases/2025-1-6-amd-announces-expanded-consumer-and-commercial-ai-.html), 2025.
[^amd-halo]: AMD, [Ryzen AI Halo developer platform specifications](https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html), accessed 2026-08-04; AMD, [Ryzen AI Max local-model sizing and bandwidth guidance](https://www.amd.com/en/blogs/2025/faqs-amd-variable-graphics-memory-vram-ai-model-sizes-quantization-mcp-more.html), 2025.
[^snapdragon-8g3]: Qualcomm, [Snapdragon 8 Gen 3 Mobile Platform](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-gen-3-mobile-platform), accessed 2026-07-30.
[^snapdragon-8-series]: Qualcomm, [Snapdragon 8 Gen 2](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-gen-2-mobile-platform), [Snapdragon 8 Gen 3](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-gen-3-mobile-platform), [Snapdragon 8 Elite](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-elite-mobile-platform), [the 8 Elite naming explanation](https://www.qualcomm.com/snapdragon/news/snapdragon-8-elite--the-power-behind-the-name), and [Snapdragon 8 Elite Gen 5](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-elite-gen-5), accessed 2026-08-04.
[^snapdragon-8-gen5]: Qualcomm, [Snapdragon 8 Gen 5 Mobile Platform](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-gen-5-mobile-platform) and [current Snapdragon 8-series catalog](https://www.qualcomm.com/smartphones/products/8-series), accessed 2026-08-04. Qualcomm compares its 46% NPU gain with Snapdragon 8 Gen 3, not Snapdragon 8 Elite.
[^snapdragon-ai-engine]: Qualcomm, [Snapdragon 855 announcement](https://www.qualcomm.com/news/releases/2018/12/qualcomm-announces-new-flagship-snapdragon-855-mobile-platform-new-decade), 2018; Qualcomm, [Snapdragon 888 announcement](https://www.qualcomm.com/news/releases/2020/12/qualcomm-redefines-premium-snapdragon-tech-summit-digital-2020), 2020; Qualcomm, [Snapdragon 888 AI technical note](https://www.qualcomm.com/media/documents/files/snapdragon-888-ai-blog-post-by-jeff-gehlhaar-vp-of-technology-hsin-i-hsu-senior-product-manager.pdf), 2020.
[^snapdragon-elite-gen5]: Qualcomm, [Snapdragon 8 Elite Gen 5](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-elite-gen-5), accessed 2026-08-04; Qualcomm, [Mobile AI platform overview](https://www.qualcomm.com/smartphones/features/mobile-ai), accessed 2026-08-04.
[^mediatek-9400]: MediaTek, [Dimensity 9400 announcement](https://corp.mediatek.com/news-events/press-releases/mediateks-dimensity-9400-flagship-soc-offers-extreme-performance-and-efficiency-for-the-latest-ai-experiences), 2024.
[^mediatek-9500]: MediaTek, [Dimensity 9500 announcement](https://www.mediatek.com/press-room/mediatek-dimensity-9500-unleashes-best-in-class-performance-ai-experiences-and-power-efficiency-for-the-next-generation-of-mobile-devices), 2025.
[^iphone-memory]: EveryMac, [iPhone model specifications](https://everymac.com/systems/apple/iphone/index-iphone-specs.html), accessed 2026-07-30.
[^iphone-x]: Apple, [The future is here: iPhone X](https://www.apple.com/newsroom/2017/09/the-future-is-here-iphone-x/), 2017.
[^iphone-xs]: Apple, [iPhone XS and iPhone XS Max](https://www.apple.com/newsroom/2018/09/iphone-xs-and-iphone-xs-max-bring-the-best-and-biggest-displays-to-iphone/), 2018.
[^iphone-11-pro]: Apple, [Apple unveils iPhone 11 Pro and iPhone 11 Pro Max](https://www.apple.com/newsroom/2019/09/apple-unveils-iphone-11-pro-and-iphone-11-pro-max/), 2019.
[^iphone-12-pro]: Apple, [Apple unveils iPhone 12 Pro and iPhone 12 Pro Max](https://www.apple.com/newsroom/2020/10/apple-unveils-iphone-12-pro-and-iphone-12-pro-max-with-5g/), 2020.
[^iphone-13-pro]: Apple, [Apple unveils iPhone 13 Pro and iPhone 13 Pro Max](https://www.apple.com/newsroom/2021/09/apple-unveils-iphone-13-pro-and-iphone-13-pro-max-more-pro-than-ever-before/), 2021.
[^phi2]: Microsoft Research, [Phi-2: The surprising power of small language models](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/), 2023.
[^llama32]: Meta, [Llama 3.2 3B model card](https://huggingface.co/meta-llama/Llama-3.2-3B), 2024.
[^phi4mini]: Microsoft, [Phi-4 Mini Instruct model card](https://huggingface.co/microsoft/Phi-4-mini-instruct), 2025.
[^llama32-quant]: Meta, [Introducing quantized Llama models](https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/), 2024.

<script defer src="{{ '/assets/vendor/chart.umd.min.js' | relative_url }}"></script>
<script defer src="{{ '/assets/chart-theme.js' | relative_url }}"></script>
<script defer src="{{ '/assets/edge-charts.js' | relative_url }}?v=20260809e"></script>
