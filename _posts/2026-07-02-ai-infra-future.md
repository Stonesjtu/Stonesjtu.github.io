---
layout: post
title: "[WIP] Predicting the future AI infra stack - AI-Infra Overview PART-3"
topic: "AI infrastructure"
sequence: 10
source_url: https://app.notion.com/p/38d2ec4bb1f0808ea061d11de43d93a6
source_label: "Original outline on Notion"
excerpt: "A supply-chain view of future AI infrastructure: heterogeneous computing, dedicated LLM hardware, memory locality, and interconnect."
---

This is the third part of the series. [Part 1](/2026/06/30/ai-infra-and-tokenomics/) defined AI infrastructure as the bridge between workloads and hardware. [Part 2](/2026/07/01/ai-infra-scaling-problem/) showed how model work, context, output, and agent loops multiply demand. [Part 4](/2026/07/03/ai-infra-edge-intelligence/) treats edge intelligence as its own hardware, economics, and model-quality problem.

This part looks forward from the supply side. The prediction is simple: **AI infrastructure will become more heterogeneous, and more hardware will be designed around LLM-specific bottlenecks rather than generic FLOPs.**

<figure class="post-figure">
  <img src="{{ '/assets/ai-infra-unified-map.svg' | relative_url }}" alt="Excalidraw systems map showing workload pressures flowing through AI infrastructure placement, compilation, caching, scheduling, runtime, and data movement into compute, memory, communication, and edge hardware constraints, with a feedback loop.">
  <figcaption>Part 3 returns to the same series map and follows its hardware side: the future stack coordinates specialization across compute, memory, communication, and edge placement.</figcaption>
</figure>

General-purpose GPUs will remain central, but the winning system increasingly looks like a coordinated package: CPUs for orchestration, GPUs for dense math, tensor engines for narrow numerical formats, NPUs for local inference, HBM and SRAM for locality, scale-up fabrics for model parallelism, scale-out networks for cluster scheduling, and software that can place work across all of them.

The evidence falls into three constraints. Compute progress increasingly depends on narrower numerical contracts. Memory determines how much model and context can stay close to arithmetic. Communication determines whether many accelerators behave like one system.

## 1. Compute becomes specialized

### Peak math is conditional

The H100 is a useful example. NVIDIA's Hopper material highlights HBM3 bandwidth around 3 TB/s, a 50 MB L2 cache, Transformer Engine support, NVLink/NVSwitch scale-out, and low-precision tensor paths.[^h100] These are not just "more FLOPs." They are area, power, and system-design choices that help specific workload shapes.

The GPU timeline shows the same pattern:

| generation | headline compute direction |
| --- | --- |
| Tesla C870 / C1060 | CUDA-era single-precision throughput |
| K20X | higher FP32/FP64 HPC throughput |
| P100 / V100 | FP16 and Tensor Core acceleration |
| A100 / H100 | BF16, TF32, FP8, sparsity, larger memory systems |
| Blackwell / Rubin | FP4/NVFP4 and rack-scale AI systems |

Representative figures make the jump visible. Tesla C870 was advertised at 518 GFLOP/s peak single precision in 2007.[^tesla-c870] C1060 reached 933 GFLOP/s in 2008.[^tesla-c1060] K20X reached 3.95 TFLOP/s single precision and 1.31 TFLOP/s double precision in 2012.[^tesla-k20x] P100 delivered 21.2 TFLOP/s FP16 in 2016, V100 delivered about 125-130 Tensor TFLOP/s, A100 reached 312 TFLOP/s dense FP16/BF16 Tensor Core performance, and H100 lists 1,979 TFLOP/s FP16/BF16 Tensor Core with sparsity, or half that without sparsity.[^p100][^v100][^a100][^h100-spec]

Blackwell continues the shift. NVIDIA's DGX B200 system lists 144 PFLOP/s FP4 Tensor Core performance across eight Blackwell GPUs, roughly 18 PFLOP/s per GPU at the published system level, while Blackwell Ultra emphasizes 15 PFLOP/s dense NVFP4 per GPU.[^blackwell-b200][^blackwell-ultra] NVIDIA's preliminary Rubin specifications list 4 PFLOP/s dense FP16/BF16 and 50 PFLOP/s NVFP4 inference compute per GPU. Vera Rubin NVL144 CPX is framed around 8 exaFLOP/s of rack-scale AI performance for massive-context inference.[^vera-rubin-spec][^rubin][^rubin-cpx]

These figures are not an apples-to-apples speedup curve. The datatype, sparsity mode, memory system, and programming model all changed. That is the important part. GPU progress came from changing the numerical contract: CUDA, SIMT execution, HBM, NVLink, tensor cores, TF32, BF16, FP8, FP4, sparsity, and compiler/runtime support made model structure visible to hardware.

### Three views of GPU economics

<details class="post-details" markdown="1">
<summary>Show normalization method and GPU price data</summary>

Peak compute, launch purchase price, and current rental price answer different questions. Keep them separate, but normalize both cost views against the same dense peak compute:

<div class="math-block">
$$
\begin{aligned}
F &= \text{peak dense FP16/BF16 PFLOP/s}, \\
L &= \frac{\text{launch system price}}{\text{GPU count}}, \\
\text{buy USD per peak PFLOP/s} &= \frac{L}{F}, \\
\text{rental USD per PFLOP-s} &= \frac{\text{USD per GPU-hour}}{3600F}
\end{aligned}
$$
</div>

Tesla C870 predates FP16 Tensor Cores, so its FP32 peak is a legacy proxy. Every later point uses dense FP16/BF16 without sparsity or lower-precision headline modes.[^h100-spec][^b200-lenovo][^vera-rubin-spec] C870 uses its standalone launch price. P100, V100, and A100 use documented DGX launch prices divided by eight GPUs. H100 and B200 use launch-window DGX-equivalent system estimates on the same per-GPU basis.[^tesla-c870-price][^dgx1-price][^gpu-launch-prices]

<details class="post-details" markdown="1">
<summary>Show launch purchase data</summary>

| GPU | normalized compute used | launch-price basis | launch cost/GPU | USD per peak PFLOP/s |
| --- | ---: | --- | ---: | ---: |
| Tesla C870 | 0.518 TFLOP/s FP32 proxy | standalone list price | USD 1,499 | USD 2.89M |
| Tesla P100 SXM2 | 21.2 TFLOP/s FP16 | DGX-1: USD 129,000 / 8 | USD 16,125 | USD 761K |
| Tesla V100 SXM | 125 TFLOP/s Tensor | DGX-1V: USD 149,000 / 8 | USD 18,625 | USD 149K |
| A100 SXM 80GB | 312 TFLOP/s FP16/BF16 Tensor | DGX A100: USD 199,000 / 8 | USD 24,875 | USD 79.7K |
| H100 SXM 80GB | 989 TFLOP/s dense FP16/BF16 Tensor | DGX-equivalent estimate: USD 269,000 / 8 | USD 33,600 | USD 34.0K |
| B200 SXM6 | 2,250 TFLOP/s dense FP16/BF16 Tensor | DGX B200 launch-window listing: USD 515,410 / 8 | USD 64,426 | USD 28.6K |

</details>

<details class="post-details" markdown="1">
<summary>Show current cloud rental data</summary>

For the current rental view, use one provider and one bundle size: Lambda's eight-GPU on-demand tier. That gives a continuous currently offered series from V100 through B200 without mixing providers or commitment discounts.[^lambda-pricing]

| GPU | dense compute | current USD/GPU-hour | rental USD per PFLOP-s |
| --- | ---: | ---: | ---: |
| Tesla V100 16GB | 0.125 PFLOP/s | USD 0.79 | USD 0.00176 |
| A100 SXM 80GB | 0.312 PFLOP/s | USD 2.79 | USD 0.00248 |
| H100 SXM 80GB | 0.989 PFLOP/s | USD 3.99 | USD 0.00112 |
| B200 SXM6 | 2.250 PFLOP/s | USD 6.69 | USD 0.000826 |

</details>

</details>

These are infrastructure proxies, not chip MSRPs or workload benchmarks. Dividing a DGX price by eight allocates CPUs, memory, storage, networking, and chassis cost to each GPU; the H100 and B200 values are launch-window estimates rather than NVIDIA-published standalone prices.[^gpu-launch-prices] Rental prices also include the provider's host system, operations, capacity, and margin. That is why the current rental curve can move differently from launch purchase economics: in Lambda's current catalog, A100 costs more per theoretical unit of dense compute than V100, then H100 and B200 resume the decline. Rubin appears only in peak compute because comparable purchase and rental prices are not yet public.

Epoch AI's broader historical work reaches the same qualitative conclusion: GPU FLOP/s per dollar doubled roughly every 2.5 years across 2006-2021, and its newer AI hardware trend page estimates AI chip performance per dollar improving by about 37% per year across 2012-2025.[^gpu-price-performance][^epoch-ai-trends] Our World in Data republishes the same broad compute-per-dollar series as an interactive chart, adjusted for inflation.[^owid-gpu-price-performance]

<figure class="post-figure post-chart">
  <div class="chart-grid">
    <section class="chart-panel">
      <p class="chart-title">Peak compute</p>
      <p class="chart-subtitle">Dense FP16/BF16 TFLOP/s where available; logarithmic scale</p>
      <div class="chart-frame chart-frame--compact"><canvas id="gpu-peak-compute-chart" role="img" aria-label="Interactive logarithmic chart showing peak dense FP16 or BF16 GPU compute from Tesla C870 through preliminary Rubin.">The source data is available in the methodology table above.</canvas></div>
    </section>
    <section class="chart-panel">
      <p class="chart-title">Historical buy cost</p>
      <p class="chart-subtitle">Release-era USD per peak PFLOP/s; lower is better</p>
      <div class="chart-frame chart-frame--compact"><canvas id="gpu-buy-cost-chart" role="img" aria-label="Interactive logarithmic chart showing release-era acquisition dollars per peak PFLOP per second falling from Tesla C870 through B200.">The source data is available in the methodology table above.</canvas></div>
    </section>
    <section class="chart-panel">
      <p class="chart-title">Current rental cost</p>
      <p class="chart-subtitle">Lambda eight-GPU on-demand USD per PFLOP-second; lower is better</p>
      <div class="chart-frame chart-frame--compact"><canvas id="gpu-rental-cost-chart" role="img" aria-label="Interactive logarithmic chart comparing current cloud rental dollars per PFLOP-second for V100, A100, H100, and B200.">The source data is available in the methodology table above.</canvas></div>
    </section>
  </div>
  <figcaption>Three distinct views share one compute basis. Peak compute uses dense FP16/BF16, except C870's FP32 legacy proxy. Historical buy cost uses release-era standalone or eight-GPU system price divided by GPU count. Current rental cost uses Lambda's eight-GPU on-demand rates accessed July 26, 2026. Lower is better in both cost plots; Rubin has a preliminary compute point but no public cost point.</figcaption>
</figure>

### ALU manufacturing: narrow math buys more lanes

The ALU-level version of the story is simpler. Arithmetic got cheaper because accelerators stopped treating every operation as a wide general-purpose floating-point operation.

<details class="post-details" markdown="1">
<summary>Show the FP16 ALU area-cost model</summary>

A lower-bound manufacturing proxy is:

<div class="math-block">
$$
\text{raw ALU cost}
\approx
\text{ALU area}
\times
\text{wafer price per mm}^2
$$
</div>

Using Horowitz's 45nm operation-area table and a 45nm 300mm wafer cost of about USD 2,000, the raw area-cost difference is already large before considering power, routing, register files, schedulers, or tensor-core reuse.[^horowitz][^alu-area-cost][^cmos-cost]

<details class="post-details" markdown="1">
<summary>Show the 45nm operation-area baseline</summary>

| operation at 45nm | area | units per mm2 | raw cost per 1M units | area advantage |
| --- | ---: | ---: | ---: | ---: |
| 16-bit FP add | 1,360 um2 | 735 | USD 38 | 3.1x vs FP32 add |
| 32-bit FP add | 4,184 um2 | 239 | USD 118 | baseline |
| 16-bit FP multiply | 1,640 um2 | 610 | USD 46 | 4.7x vs FP32 multiply |
| 32-bit FP multiply | 7,700 um2 | 130 | USD 218 | baseline |

</details>

This is the silicon reason lower-precision tensor paths can improve compute per dollar. If a workload tolerates FP16, BF16, FP8, FP4, sparsity, or structured matrix engines, the chip can spend the same die area on many more arithmetic lanes. The catch is that those lanes only become useful when the model, compiler, kernels, and memory system keep them fed.

For a rough 20-year shrinkage estimate, take the 45nm 16-bit FP add plus 16-bit FP multiply area as the baseline:

<div class="math-block">
$$
A_{\text{FP16 mul+add},45\text{nm}}
\approx
1{,}360 + 1{,}640
= 3{,}000\ \mu m^2
$$
</div>

Then scale that logical datapath by public logic-density estimates. This is not a real vendor tensor-core layout. It is a normalized "same logic, denser process" estimate. The density anchors use 28/16/7nm TSMC comparisons, 5nm process-node density data, 3nm process-node density data, and next-node N2 / 18A estimates. Wafer prices reuse the same public wafer-price anchors used above; the next-node row uses a USD 30,000 wafer proxy from public 2nm pricing reports.[^logic-density-28-7][^process-density-5nm][^process-density-3nm][^next-node-density][^cset-wafer-cost][^wafer-pricing][^n2-wafer-price]

N+1, N+2, and N+3 are **SMIC**, not TSMC, process names. They form a separate 7nm-class branch confirmed through product teardowns. Public N+1 density is not sufficiently documented for this model. For N+2, I infer about 92.9 MTr/mm2 from SemiAnalysis measurements showing N+3's cell height and contacted-gate pitch each shrinking 9.5% from N+2. N+3 itself measures 113.4 MTr/mm2.[^smic-n1-n2][^smic-n3]

<details class="post-details" markdown="1">
<summary>Show the process-node ALU model</summary>

| node | era | logic-density reference | estimated FP16 mul+add area | area shrink vs 45nm | units per mm2 | raw cost per 1M units |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 45nm | 2007 | 6.25 MTr/mm2 baseline | 3,000 um2 | 1.0x | 333 | USD 85 |
| 28nm | 2010 | 15.3 MTr/mm2 | 1,225 um2 | 2.4x | 816 | USD 52 |
| 16nm | 2015 | 28.9 MTr/mm2 | 649 um2 | 4.6x | 1,541 | USD 37 |
| 7nm | 2018 | 91.2 MTr/mm2 | 206 um2 | 14.6x | 4,864 | USD 27 |
| 5nm | 2020 | 138.2 MTr/mm2 | 136 um2 | 22.1x | 7,371 | USD 33 |
| SMIC N+1, 7nm-class | 2021-2022 | teardown confirmed; density not used | not modeled | not modeled | not modeled | wafer price not public |
| SMIC N+2, 7nm-class | 2023 | about 92.9 MTr/mm2, inferred | 202 um2 | 14.9x | 4,955 | wafer price not public |
| 3nm | 2024 | 216 MTr/mm2 | 87 um2 | 34.6x | 11,520 | USD 24 |
| SMIC N+3, 7nm-class | 2025 | 113.4 MTr/mm2, teardown | 165 um2 | 18.1x | 6,048 | wafer price not public |
| N2 estimate | 2026 | 313 MTr/mm2 | 60 um2 | 50.1x | 16,693 | USD 25 |
| 18A estimate | 2026 | 238 MTr/mm2 | 79 um2 | 38.1x | 12,693 | USD 33 |
| Huawei LogicFolding roadmap | 2026 onward | 3D/package-footprint density; not a node | not comparable | not comparable | not comparable | not modeled |

</details>

</details>

The area trend is the important signal: a 16-bit floating-point multiply-plus-add datapath that is about 3,000 um2 at 45nm becomes an order of magnitude smaller by 7nm, roughly 35x smaller by 3nm, and about 50x smaller at an N2-style next node under a pure logic-density scaling model. SMIC's N+1 to N+3 branch shows how DUV multi-patterning and design-technology co-optimization can keep improving a 7nm-class platform, but SemiAnalysis finds that N+3 pays for its TSMC N6-class density in process complexity, efficiency, and control.[^smic-n3]

Huawei LogicFolding is a different scaling axis. Huawei describes it as shortening critical-path wiring, increasing density, and reaching its first commercial Kirin implementation in fall 2026; SemiAnalysis characterizes the approach as stacking active logic and recovering density through advanced packaging.[^huawei-logic-folding][^smic-n3] It should not be converted into a smaller planar ALU area: package-footprint density can rise by stacking active layers even when each underlying die remains on a less-dense process.

The raw wafer-cost proxy falls less smoothly because advanced wafer prices rise sharply. N2 can fit more arithmetic, but a USD 30,000 wafer can erase much of the dollar-per-ALU gain versus 3nm. Real ALUs also need registers, operand routing, control, clocking, SRAM, verification margin, and yield. Tensor cores improve the economics further by amortizing control and data movement across matrix tiles instead of treating every multiply-add as an isolated scalar unit.

<figure class="post-figure post-chart">
  <section class="chart-panel">
    <p class="chart-title">Estimated raw FP16 ALU dollar cost</p>
    <p class="chart-subtitle">USD per one million multiply-plus-add datapaths; logarithmic scale</p>
    <div class="chart-frame"><canvas id="alu-cost-chart" role="img" aria-label="Interactive logarithmic chart showing the estimated raw cost of one million FP16 multiply-plus-add datapaths from 45 nanometer through N2 and Intel 18A.">The process-node estimates are available in the table above.</canvas></div>
  </section>
  <figcaption>Logic density keeps shrinking the raw FP16 arithmetic datapath, but wafer prices flatten the dollar-cost curve at leading-edge nodes.</figcaption>
</figure>

## 2. Memory sets the working set

Memory and manufacturing show the same pattern. Compute can keep rising, but every token also needs bytes close to the math unit. The difficult part is that each level of memory optimizes a different constraint: on-chip SRAM is fast but area-expensive, HBM is bandwidth-rich but package-expensive, commodity DRAM is capacity-rich but far away, and advanced wafers are no longer getting cheap fast enough to hide the tradeoff.

<details class="post-details" markdown="1">
<summary>Show SRAM, HBM, DRAM, and wafer methodology</summary>

### On-chip SRAM: fast bytes are area-limited

For on-chip SRAM, there is no public spot price per MB. A useful lower-bound proxy is:

<div class="math-block">
$$
\text{raw SRAM cost per MB}
\approx
\text{SRAM bitcell area per MB}
\times
\text{wafer price per mm}^2
$$
</div>

This ignores periphery, redundancy, yield, cache tags, routing, and design cost, so it is not a product cost. It is still useful because it shows why cache capacity is no longer free. TSMC reported a 0.127 um2 28nm 6T SRAM cell in 2009; public 5nm and 2nm SRAM reports put high-density bitcells around 0.021 um2; CSET estimated 7nm and 5nm wafer sale prices at USD 9,346 and USD 16,988 respectively; public 2026 wafer-price roundups put TSMC 3nm around USD 19,500.[^tsmc-28nm-sram][^tsmc-5nm-sram][^tsmc-2nm-sram][^cset-wafer-cost][^wafer-pricing]

<details class="post-details" markdown="1">
<summary>Show the SRAM area-cost proxy</summary>

| node | SRAM bitcell | wafer price used | raw MB area | raw SRAM cost proxy |
| --- | ---: | ---: | ---: | ---: |
| 28nm | 0.127 um2 | USD 3,000 | 1.065 mm2/MB | USD 0.045/MB |
| 7nm | 0.027 um2 | USD 9,346 | 0.226 mm2/MB | USD 0.030/MB |
| 5nm | 0.021 um2 | USD 16,988 | 0.176 mm2/MB | USD 0.042/MB |
| 3nm / 2nm-class | 0.021 um2 | USD 19,500 | 0.176 mm2/MB | USD 0.049/MB |

</details>

The punchline is not that SRAM got worse in absolute density. It got much denser. The punchline is that after 7nm, bitcell shrink is small while wafer cost keeps rising. That means larger on-chip caches require more deliberate architectural justification: more L2, more shared memory, tensor memory, larger register files, and better reuse have to earn their silicon area.

NVIDIA GPU caches show the architectural response. P100 had about 4 MB of L2, V100 6 MB, A100 40 MB, H100 50 MB, and public B200 analysis reports about 126 MB of total L2.[^a100][^h100][^chips-b200-cache] More on-chip SRAM is being used because going to HBM is expensive in energy and latency, but the amount is still tiny compared with model state and KV cache.

### Off-chip memory: capacity and bandwidth diverge

Off-chip memory has split into two worlds. Commodity DRAM remains the capacity workhorse, but its price-per-GB improvement slowed sharply after 2010. Stanford DAM's compiled memory-price dataset shows cheapest DRAM falling from about USD 185/GB in 2005 to USD 12.2/GB in 2010, then only to USD 3.0/GB by 2020 and about USD 3.45/GB in July 2026.[^stanford-memory-prices] HBM moves in the other direction: it is not cheap capacity, it is purchased bandwidth close to the accelerator. Rambus summarizes HBM's speed evolution from 128 GB/s per HBM device to 2.048 TB/s for HBM4, while Stanford DAM's modeled HBM data puts HBM2e around USD 6/GB and HBM3e peak around USD 18/GB.[^rambus-hbm][^stanford-memory-prices]

Vera Rubin makes the next step concrete. NVIDIA's preliminary specification assigns each Rubin GPU 288 GB of HBM4 with 22 TB/s of memory bandwidth. That is 1.6x B200's 180 GB capacity and about 2.9x its 7.7 TB/s bandwidth. At NVL72 scale, 72 Rubin GPUs aggregate to 20.7 TB of HBM4 and 1,580 TB/s of memory bandwidth.[^vera-rubin-spec]

<details class="post-details" markdown="1">
<summary>Show off-chip capacity and bandwidth anchors</summary>

| off-chip memory trend | representative anchors |
| --- | --- |
| GPU HBM generation | P100/V100: HBM2; A100 40GB: HBM2; A100 80GB: HBM2e; H100: HBM3; B200: HBM3e; Rubin: HBM4[^p100][^v100][^a100][^h100][^b200-lenovo][^vera-rubin-spec] |
| GPU HBM capacity | P100: 16 GB; V100: 16 GB; A100: 40 GB; H100: 80 GB; B200: 180 GB; Rubin: 288 GB[^p100][^v100][^a100][^h100][^b200-lenovo][^vera-rubin-spec] |
| GPU HBM bandwidth | P100: 720 GB/s; V100: 900 GB/s; A100: 1,555 GB/s; H100: over 3 TB/s; B200: 7.7 TB/s; Rubin: 22 TB/s[^p100][^v100][^a100][^h100][^b200-lenovo][^vera-rubin-spec] |
| Vera Rubin NVL72 aggregate | 20.7 TB HBM4 capacity and 1,580 TB/s memory bandwidth across 72 Rubin GPUs[^vera-rubin-spec] |
| commodity DRAM price/capacity | about USD 185/GB in 2005, USD 12.2/GB in 2010, USD 3.0/GB in 2020, and USD 3.45/GB in July 2026[^stanford-memory-prices] |
| HBM price/capacity | HBM2e around USD 6/GB, HBM3 around USD 9/GB, HBM3e peak around USD 18/GB, HBM4 projected around USD 16.5/GB[^stanford-memory-prices] |
| HBM price/bandwidth | HBM2e around USD 209 per TB/s, HBM3 around USD 264 per TB/s, HBM3e peak around USD 352 per TB/s, HBM4 projected around USD 297 per TB/s[^stanford-memory-prices] |

</details>

### Manufacturing: wafer cost pushes back

The manufacturing layer is the shared denominator under both compute and SRAM. If each wafer gets more expensive, every large die, cache expansion, interposer choice, and yield loss has a higher dollar impact.

<details class="post-details" markdown="1">
<summary>Show advanced wafer-price anchors</summary>

| process node | approximate era | wafer price anchor | why it matters |
| --- | ---: | ---: | --- |
| 28nm | 2010 | USD 3,000 | cheap enough that SRAM scaling still translated into lower raw MB cost |
| 7nm | 2018 | USD 9,346 | density improved, but wafer price more than tripled |
| 5nm | 2020 | USD 16,988 | raw SRAM cost proxy rose again despite smaller cells |
| 3nm | 2024 | USD 19,500 | wafer price keeps rising while SRAM bitcell shrink slows |

</details>

</details>

<figure class="post-figure post-chart">
  <div class="chart-grid chart-grid--two">
    <section class="chart-panel">
      <p class="chart-title">On-chip SRAM cost proxy</p>
      <p class="chart-subtitle">Raw bitcell USD/MB lower bound</p>
      <div class="chart-frame chart-frame--compact"><canvas id="sram-cost-chart" role="img" aria-label="Interactive chart showing the raw SRAM bitcell cost proxy across 28 nanometer, 7 nanometer, 5 nanometer, and 3 nanometer or N2-class nodes.">The methodology is available above.</canvas></div>
    </section>
    <section class="chart-panel">
      <p class="chart-title">Commodity DRAM price</p>
      <p class="chart-subtitle">Cheapest listed USD/GB; logarithmic scale</p>
      <div class="chart-frame chart-frame--compact"><canvas id="dram-cost-chart" role="img" aria-label="Interactive logarithmic chart showing commodity DRAM dollars per gigabyte falling from 2005 through 2026.">The methodology is available above.</canvas></div>
    </section>
    <section class="chart-panel">
      <p class="chart-title">HBM bandwidth price</p>
      <p class="chart-subtitle">Modeled USD per TB/s</p>
      <div class="chart-frame chart-frame--compact"><canvas id="hbm-bandwidth-cost-chart" role="img" aria-label="Interactive chart showing modeled HBM dollars per terabyte per second for HBM2e through projected HBM4.">The methodology is available above.</canvas></div>
    </section>
    <section class="chart-panel">
      <p class="chart-title">Advanced wafer price</p>
      <p class="chart-subtitle">Foundry sale price per 300 mm wafer; logarithmic scale</p>
      <div class="chart-frame chart-frame--compact"><canvas id="wafer-price-chart" role="img" aria-label="Interactive logarithmic chart showing approximate advanced wafer prices across 28 nanometer through 3 nanometer.">The methodology is available above.</canvas></div>
    </section>
  </div>
  <figcaption>Memory economics explain why AI infra is increasingly about locality: on-chip SRAM density is harder to buy with node shrinks, HBM bandwidth is expensive capacity, commodity DRAM is cheap but far away, and advanced wafer prices keep rising.</figcaption>
</figure>

## 3. Communication becomes topology

The next bottleneck appears when one accelerator is not enough. Scaling out turns compute into a distributed system problem: GPUs must exchange gradients, activations, KV cache state, expert routes, pipeline bubbles, and scheduling metadata. A useful first-order model is the latency-bandwidth model:

<div class="math-block">
$$
T_{\text{comm}}
\approx
\alpha \cdot n_{\text{messages}}
+
\frac{\text{bytes moved}}{B_{\text{effective}}}
$$
</div>

Here `alpha` is the per-message latency cost and `B_effective` is achieved communication bandwidth after topology, protocol, contention, and collective implementation overhead. That term matters because modern training and inference are full of collectives:

<details class="post-details" markdown="1">
<summary>Show parallelism and scale-up bandwidth anchors</summary>

<details class="post-details" markdown="1">
<summary>Show communication pressure by parallelism pattern</summary>

| parallelism pattern | communication pressure |
| --- | --- |
| data parallel | gradient all-reduce or reduce-scatter / all-gather |
| tensor parallel | activation all-reduce and all-gather inside layers |
| pipeline parallel | boundary activations and pipeline bubbles |
| expert parallel / MoE | token dispatch and all-to-all routing |
| disaggregated serving | KV cache movement, prefill/decode handoff, remote memory access |

</details>

Interconnect bandwidth is improving aggressively because this tax is now first-order. NVIDIA lists NVLink bandwidth per GPU rising from 900 GB/s on Hopper to 1.8 TB/s on Blackwell and 3.6 TB/s on Rubin; its NVLink Switch table lists NVL72 aggregate bandwidth rising from 130 TB/s on Blackwell to 260 TB/s on Rubin.[^nvlink] NVIDIA's HGX Rubin page makes the same point at the system level: higher token throughput is tied not only to more NVFP4 compute, but also to more HBM bandwidth and more NVLink Switch bandwidth.[^hgx-rubin]

<details class="post-details" markdown="1">
<summary>Show Hopper, Blackwell, and Rubin scale-up anchors</summary>

| system generation | interconnect anchor | why it matters |
| --- | ---: | --- |
| Hopper | 900 GB/s NVLink per GPU | scale-up communication becomes part of model throughput |
| Blackwell | 1.8 TB/s NVLink per GPU; 130 TB/s NVL72 aggregate | larger rack-scale GPU domains for model parallelism |
| Rubin | 3.6 TB/s NVLink per GPU; 260 TB/s NVL72 aggregate | communication bandwidth has to scale with MoE, long context, and agentic inference |

</details>

</details>

There are two different network curves hiding behind the same word "interconnect":

- **Scale-up** is the tightly coupled GPU domain inside a box or rack. NVLink / NVSwitch bandwidth is not sold like a generic switch port; it is bundled into GPU systems, board design, power delivery, and thermal design. The useful public metric is bandwidth per GPU or per rack-scale domain.
- **Scale-out** is the cluster fabric across nodes and racks. Ethernet and InfiniBand have visible port speeds, switch radix, optics, cables, NICs, and sometimes observable street prices. This is where a rough dollar-per-Gb/s proxy is possible.

<figure class="post-figure">
  <img src="{{ '/assets/scale-up-scale-out-excalidraw.svg' | relative_url }}" alt="Excalidraw topology showing GPUs connected by NVSwitch inside two rack-scale domains, with the domains joined by an Ethernet or InfiniBand scale-out fabric.">
  <figcaption>Scale-up tries to make accelerators behave like one machine; scale-out connects many machines while exposing more latency, topology, and software-visible coordination.</figcaption>
</figure>

The speed curve is steep. InfiniBand moved from 4x QDR at 32 Gb/s in the late 2000s, to EDR 100 Gb/s, HDR 200 Gb/s, NDR 400 Gb/s, and XDR 800 Gb/s. Ethernet followed the same broad shape: 40/100GbE was standardized in 2010, 200/400GbE in 2017, and 800GbE in 2024.[^infiniband-rates][^ethernet-100g][^ethernet-400g][^ethernet-800g] NVIDIA's current Quantum-X800 documentation lists 72-port and 144-port XDR systems at 800 Gb/s per port, up to 115.2 Tb/s of maximum throughput for the 4U system.[^quantum-x800]

<details class="post-details" markdown="1">
<summary>Show scale-out price methodology and switch data</summary>

For a dollar-per-speed proxy, use switch chassis price divided by front-panel bandwidth. This is not total cluster networking cost. It excludes optics, cables, NICs, support contracts, power, rack layout, and topology oversubscription. It is still useful because it shows the direction of the switching layer itself. The anchors below combine a legacy Cisco 10GbE price-list snapshot, public Mellanox / NVIDIA InfiniBand switch listings, and current SN5610 800GbE listing/spec data.[^nexus-price][^mellanox-switch-prices][^sn5610-price][^sn5610-specs]

<details class="post-details" markdown="1">
<summary>Show scale-out switch price anchors</summary>

| scale-out switch proxy | approximate era | ports x port speed | public price anchor | switch dollars per Gb/s |
| --- | ---: | ---: | ---: | ---: |
| Cisco Nexus 5020 10GbE | 2008 | 40 x 10 Gb/s | USD 28,770 list-price snapshot | USD 72/Gb/s |
| Mellanox SB7800 EDR | 2015 | 36 x 100 Gb/s | USD 10,259 channel listing | USD 2.85/Gb/s |
| Mellanox QM8700 HDR | 2018 | 40 x 200 Gb/s | USD 18,740 channel listing | USD 2.34/Gb/s |
| NVIDIA QM9700 NDR | 2022 | 64 x 400 Gb/s | USD 32,870 channel listing | USD 1.28/Gb/s |
| NVIDIA SN5610 800GbE | 2026 | 64 x 800 Gb/s | USD 51,999 channel listing | USD 1.02/Gb/s |

</details>

</details>

<figure class="post-figure post-chart">
  <div class="chart-grid chart-grid--two">
    <section class="chart-panel">
      <p class="chart-title">Scale-out port speed</p>
      <p class="chart-subtitle">Front-panel Gb/s; logarithmic scale</p>
      <div class="chart-frame chart-frame--compact"><canvas id="interconnect-speed-chart" role="img" aria-label="Interactive logarithmic chart showing scale-out port speed increasing from 10 gigabits per second in 2008 to 800 gigabits per second in 2026.">The switch data is available in the table above.</canvas></div>
    </section>
    <section class="chart-panel">
      <p class="chart-title">Switch cost per bandwidth</p>
      <p class="chart-subtitle">Chassis-only USD/Gb/s proxy; logarithmic scale</p>
      <div class="chart-frame chart-frame--compact"><canvas id="interconnect-cost-chart" role="img" aria-label="Interactive logarithmic chart showing switch dollars per gigabit per second falling from about 72 dollars to about one dollar.">The switch data is available in the table above.</canvas></div>
    </section>
  </div>
  <figcaption>Scale-out switch bandwidth improved dramatically over the last two decades, but the system-level network bill does not fall as fast as raw switch dollars per Gb/s because optics, NICs, cables, power, and topology complexity become first-order costs.</figcaption>
</figure>

The punchline is subtle: network silicon has delivered a large cost-per-bit improvement, but AI clusters keep spending the savings. Higher port speed enables larger all-reduce domains, more tensor-parallel shards, more MoE all-to-all traffic, and more disaggregated serving. The value of better AI infrastructure is therefore not just "buy faster switches." It is reducing bytes moved, placing bytes closer to compute, overlapping collectives with kernels, and choosing parallelism plans that turn expensive network bandwidth into useful tokens.

This is why "chips are slowing down" is not only a FLOP story. It is a locality and communication story. When model weights, activations, KV cache, and tool-use context grow, the system pays for bytes in several currencies: SRAM area, HBM dollars, HBM bandwidth, interconnect bandwidth, synchronization time, package complexity, wafer cost, and energy. Good AI infrastructure wins by spending fewer bytes, reusing them closer to compute, and making expensive memory and network bandwidth do useful work more often.

## Edge becomes a separate placement frontier

Edge hardware changes the accounting unit. A phone, AI PC, or robotics box is bounded by shared memory, bandwidth, thermals, battery life, and a product-level price rather than by rack power and HBM alone. Peak TOPS is useful, but it does not tell us how large a model fits, how quickly its weights can be streamed during decoding, or how much benchmark quality survives quantization.

[Part 4 follows this edge intelligence envelope directly](/2026/07/03/ai-infra-edge-intelligence/): how mobile, AI PC, and AI-box hardware evolved; how their product-level dollars per advertised performance changed; how edge-fit language models improved; and what those trends imply for local agents.

## Prediction: heterogeneity becomes the default

The accounting unit connects the evidence:

```text
token cost ~= math + memory movement + communication + scheduling + retries
```

The next AI infra frontier is not simply a larger cluster. It is a tighter control loop between algorithms, model architecture, serving systems, kernels, compilers, memory hierarchy, interconnect, edge devices, and chips.

The practical direction is heterogeneous and increasingly LLM-specific:

- More dedicated tensor formats and matrix engines, because narrow numerical contracts buy more useful math per watt and per dollar.
- More memory-aware architectures, because context, KV cache, retrieval, and multimodal state make bytes as important as FLOPs.
- More rack-scale and cluster-scale co-design, because scale-up and scale-out communication now shape model design.
- More edge/cloud partitioning, because [not every inference should cross the network and not every local device can host the whole model](/2026/07/03/ai-infra-edge-intelligence/).
- More compiler/runtime responsibility, because specialized hardware only matters when the software stack can expose locality, regularity, and parallelism.

The teams that win will not only have better models or better hardware. They will have better translation between the two: less work per useful result, better placement for every stage, and higher utilization of every expensive byte and arithmetic lane.

## References

[^h100]: NVIDIA, [Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/), 2022.
[^tesla-c870]: NVIDIA, [NVIDIA Tesla: GPU Compute Technical Brief](https://www.nvidia.com/docs/io/43395/tesla_technical_brief.pdf), 2007.
[^tesla-c1060]: NVIDIA, [Tesla C1060 Computing Processor Board](https://www.nvidia.com/docs/io/43395/bd-04111-001_v06.pdf), 2008.
[^tesla-k20x]: NVIDIA Newsroom, [NVIDIA Unveils World's Fastest, Most Efficient Accelerators](https://nvidianews.nvidia.com/news/nvidia-unveils-world-s-fastest-most-efficient-accelerators-powers-world-s-no-1-supercomputer-6622729), 2012.
[^p100]: NVIDIA, [Pascal Architecture Whitepaper](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf), 2016.
[^v100]: NVIDIA, [Tesla V100 GPU Architecture](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf), 2017.
[^a100]: NVIDIA, [A100 Tensor Core GPU Architecture](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf), 2020.
[^h100-spec]: NVIDIA, [H100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h100/), accessed 2026-07-02.
[^blackwell-b200]: NVIDIA, [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/), accessed 2026-07-02.
[^blackwell-ultra]: NVIDIA Developer Blog, [Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/), 2026.
[^vera-rubin-spec]: NVIDIA, [Vera Rubin NVL72](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/), preliminary specifications accessed 2026-07-26.
[^rubin]: NVIDIA Newsroom, [NVIDIA Kicks Off the Next Generation of AI With Rubin](https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer), 2026.
[^rubin-cpx]: NVIDIA Newsroom, [NVIDIA Unveils Rubin CPX](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference), 2025.
[^b200-lenovo]: Lenovo Press, [ThinkSystem NVIDIA HGX B200 180GB 1000W GPU](https://lenovopress.lenovo.com/lp2226-thinksystem-nvidia-b200-180gb-1000w-gpu), accessed 2026-07-04.
[^tesla-c870-price]: Michael Feldman, [NVIDIA Takes Direct Aim at High Performance Computing](https://www.hpcwire.com/2007/06/22/nvidia_takes_direct_aim_at_high_performance_computing-1/), HPCwire, 2007.
[^dgx1-price]: NVIDIA, [NVIDIA DGX-1](https://www.nvidia.com/en-au/data-center/dgx-1/), listing USD 129,000 for the eight-P100 system and USD 149,000 for the eight-V100 system.
[^gpu-launch-prices]: CCIR Research, [Rent and MSRP: Five Generations of Posted Prices](https://ccir.io/research/rent-and-msrp), 2026. The dataset derives per-GPU launch-window prices from eight-GPU system prices and grades the H100 and B200 estimates as vendor-adjacent rather than official standalone MSRP.
[^lambda-pricing]: Lambda, [GPU Instances](https://lambda.ai/instances), eight-GPU on-demand price per GPU-hour, accessed 2026-07-26.
[^gpu-price-performance]: Jaime Sevilla and Pablo Villalobos, [Trends in GPU Price-Performance](https://epoch.ai/publications/trends-in-gpu-price-performance), Epoch AI, 2022.
[^epoch-ai-trends]: Epoch AI, [Trends in Artificial Intelligence: AI Hardware](https://epoch.ai/trends), accessed 2026-07-02.
[^owid-gpu-price-performance]: Our World in Data, [GPU computational performance per dollar](https://ourworldindata.org/grapher/gpu-price-performance), accessed 2026-07-02.
[^horowitz]: Mark Horowitz, [Computing's Energy Problem](https://gwern.net/doc/cs/hardware/2014-horowitz-2.pdf), ISSCC 2014.
[^alu-area-cost]: Ting-Yu Yeh, [Accelerator Architectures for Machine Learning](https://people.cs.nycu.edu.tw/~ttyeh/course/2023_Fall/IOC5009/slide/lecture-3.pdf), lecture slides citing Horowitz ISSCC 2014 operation energy and area data, accessed 2026-07-04.
[^cmos-cost]: Tim Johnson, [CMOS Cost](https://faculty-web.msoe.edu/johnsontimoj/EE4980/files4980/cmos_cost.pdf), MSOE EE 4980 notes, accessed 2026-07-04.
[^logic-density-28-7]: Team VLSI, [TSMC 7nm, 16nm and 28nm Technology node comparisons](https://teamvlsi.com/2021/09/tsmc-7nm-16nm-and-28nm-technology-node-comparisons.html), 2021.
[^process-density-5nm]: Wikipedia, [5 nm process](https://en.wikipedia.org/wiki/5_nm_process), accessed 2026-07-04.
[^process-density-3nm]: Wikipedia, [3 nm process](https://en.wikipedia.org/wiki/3_nm_process), accessed 2026-07-04.
[^next-node-density]: Anton Shilov, [Intel's 18A and TSMC's N2 process nodes compared](https://www.tomshardware.com/tech-industry/intels-18a-and-tsmcs-n2-process-nodes-compared-intel-is-faster-but-tsmc-is-denser), Tom's Hardware, 2025.
[^smic-n1-n2]: TechInsights, [Confirming SMIC N+2 7nm in Huawei Mate 60 Pro](https://www.techinsights.com/blog/techinsights-confirming-smic-n2-7nm-huawei-mate-60-pro), documenting N+1 in 2022 and the commercial N+2 generation in 2023.
[^smic-n3]: SemiAnalysis STEEL Team et al., [Is SMIC N+3's Metal Pitch Smaller than Intel 18A's?](https://newsletter.semianalysis.com/p/steel-smic-n3-teardown), 2026.
[^huawei-logic-folding]: Huawei, [Huawei Presents the Tau Scaling Law](https://www.huawei.com/en/news/2026/5/ieee-iscas-tau-scaling), 2026.
[^cset-wafer-cost]: Center for Security and Emerging Technology, [Analysts believe that a single TSMC 5nm wafer costs USD 17,000](https://cset.georgetown.edu/article/analysts-believe-that-a-single-tsmc-5nm-wafer-costs-17000/), 2020.
[^wafer-pricing]: Silicon Analysts, [Semiconductor Wafer Pricing by Process Node](https://siliconanalysts.com/data/wafer-pricing), accessed 2026-07-04.
[^n2-wafer-price]: Astute Group, [TSMC's 2nm Wafer Price Hits USD 30,000 Amid Monopoly Concerns](https://www.astutegroup.com/news/industrial/tsmcs-2nm-wafer-price-hits-30000-amid-monopoly-concerns/), 2025.
[^tsmc-28nm-sram]: Mark LaPedus, [TSMC devises SRAM cell at 28-nm](https://www.eetimes.com/tsmc-devises-sram-cell-at-28-nm/), EE Times, 2009.
[^tsmc-5nm-sram]: SemiWiki, [TSMC's 5nm 0.021um2 SRAM Cell Using EUV and High Mobility Channel with Write Assist at ISSCC2020](https://semiwiki.com/semiconductor-manufacturers/tsmc/283487-tsmcs-5nm-0-021um2-sram-cell-using-euv-and-high-mobility-channel-with-write-assist-at-isscc2020/), 2020.
[^tsmc-2nm-sram]: TSMC Research, [Memory publications](https://research.tsmc.com/english/research/memory/publish-time-1.html), accessed 2026-07-04.
[^chips-b200-cache]: Chips and Cheese, [Nvidia's B200: Keeping the CUDA Juggernaut Rolling](https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut), 2025.
[^stanford-memory-prices]: David Shim, Stanford DAM, [Memory Prices](https://dam.stanford.edu/memory-prices.html), accessed 2026-07-04.
[^rambus-hbm]: Rambus, [High Bandwidth Memory: Everything You Need to Know](https://www.rambus.com/blogs/hbm3-everything-you-need-to-know/), updated 2026.
[^nvlink]: NVIDIA, [NVLink and NVLink Switch](https://www.nvidia.com/en-us/data-center/nvlink/), accessed 2026-07-04.
[^hgx-rubin]: NVIDIA, [HGX Platform](https://www.nvidia.com/en-us/data-center/hgx/), accessed 2026-07-04.
[^infiniband-rates]: Wikipedia, [InfiniBand performance table](https://en.wikipedia.org/wiki/InfiniBand#Performance), accessed 2026-07-04.
[^ethernet-100g]: Wikipedia, [100 Gigabit Ethernet](https://en.wikipedia.org/wiki/100_Gigabit_Ethernet), accessed 2026-07-04.
[^ethernet-400g]: Ethernet Alliance, [IEEE 802.3 Standards Activities](https://ethernetalliance.org/wp-content/uploads/2018/02/OFC_400G_18_0314_Final.pdf), 2018.
[^ethernet-800g]: IEEE Standards Association, [Ethernet's Next Bar is Now - 800 Gb/s!](https://standards.ieee.org/beyond-standards/ethernets-next-bar/), 2024.
[^quantum-x800]: NVIDIA Networking Docs, [NVIDIA Q32xx and Q34xx XDR 800Gb/s InfiniBand Switch Systems](https://networking-docs.nvidia.com/xdrswitcheshw/introduction), accessed 2026-07-04.
[^nexus-price]: Finnegan Software, [Cisco price list snapshot](https://www.finnsoft.com/priclist/cisco.htm), accessed 2026-07-04.
[^mellanox-switch-prices]: Router-Switch.com, [NVIDIA Mellanox switches price list](https://www.router-switch.com/mellanox-switches-price.html), accessed 2026-07-04.
[^sn5610-price]: NADDOD, [NVIDIA SN5610 Spectrum-4 800GbE switch listing](https://www.naddod.com/products/102969.html), accessed 2026-07-04.
[^sn5610-specs]: NVIDIA Networking Docs, [NVIDIA Spectrum-4 SN5000 specifications](https://docs.nvidia.com/networking/display/sn5000/specifications), accessed 2026-07-04.

<script defer src="{{ '/assets/vendor/chart.umd.min.js' | relative_url }}"></script>
<script defer src="{{ '/assets/chart-theme.js' | relative_url }}"></script>
<script defer src="{{ '/assets/series-charts.js' | relative_url }}"></script>
