---
layout: post
title: "Predicting the future AI infra stack - AI-Infra Overview PART-3"
description: "A supply-chain view of future AI infrastructure: heterogeneous compute, LLM-specific hardware, memory locality, and high-bandwidth interconnects."
topic: "AI infrastructure"
sequence: 10
last_modified_at: 2026-08-10T22:25:18+08:00
source_url: https://app.notion.com/p/38d2ec4bb1f0808ea061d11de43d93a6
source_label: "Original outline on Notion"
excerpt: "A supply-chain view of future AI infrastructure: heterogeneous computing, dedicated LLM hardware, memory locality, and interconnect."
---

This is the third part of the series. [Part 1](/2026/06/30/ai-infra-and-tokenomics/) defined AI infrastructure as the implementation layer that maps models onto hardware under a service objective. [Part 2](/2026/07/01/ai-infra-scaling-problem/) showed how model work, context, output, and agent loops multiply demand. [Part 4](/2026/07/03/ai-infra-edge-intelligence/) treats edge intelligence as its own hardware, economics, and model-quality problem.

This part looks forward from the supply side.

<p class="key-insight"><strong>Key insight</strong><span>AI infrastructure will become more heterogeneous, and more hardware will be designed around LLM-specific bottlenecks rather than generic FLOPs.</span></p>

<figure class="post-figure">
  <img src="{{ '/assets/ai-infra-unified-map.svg' | relative_url }}?v=20260809" alt="Excalidraw framework titled AI Infra connects Model to Hardware. Model computation, state, and execution mode flow through infrastructure compilation, placement, caching, scheduling, and operations onto hardware compute engines, memory hierarchy, topology, and power under a service objective.">
  <figcaption>Part 3 returns to the same framework and follows its hardware side: changes in compute engines, memory hierarchy, topology, and power reshape the infrastructure mapping and eventually feed back into model design.</figcaption>
</figure>

General-purpose GPUs will remain central, but the winning system increasingly looks like a coordinated package: CPUs for orchestration, GPUs for dense math, tensor engines for narrow numerical formats, NPUs for local inference, HBM and SRAM for locality, scale-up fabrics for model parallelism, scale-out networks for cluster scheduling, and software that can place work across all of them.

The evidence falls into three constraints. Compute progress increasingly depends on narrower numerical contracts. Memory determines how much model and context can stay close to arithmetic. Communication determines whether many accelerators behave like one system.

## 1. Compute becomes specialized

<p class="key-insight"><strong>Key insight</strong><span>Peak dense compute keeps rising and dollars per peak compute keep falling, but the gains increasingly come from narrower numerical contracts and workload-specific engines rather than transistor shrink alone.</span></p>

<figure class="post-figure post-chart">
  <div class="chart-grid">
    <section class="chart-panel">
      <p class="chart-title">Peak compute</p>
      <p class="chart-subtitle">Dense FP16/BF16 TFLOP/s where available; logarithmic scale</p>
      <div class="chart-frame chart-frame--compact"><canvas id="gpu-peak-compute-chart" role="img" aria-label="Interactive logarithmic chart showing peak dense FP16 or BF16 GPU compute from Tesla C870 through preliminary Rubin.">The source data is available in the methodology table below.</canvas></div>
    </section>
    <section class="chart-panel">
      <p class="chart-title">Historical buy cost</p>
      <p class="chart-subtitle">Release-era USD per peak PFLOP/s; lower is better</p>
      <div class="chart-frame chart-frame--compact"><canvas id="gpu-buy-cost-chart" role="img" aria-label="Interactive logarithmic chart showing release-era acquisition dollars per peak PFLOP per second falling from Tesla C870 through B200.">The source data is available in the methodology table below.</canvas></div>
    </section>
    <section class="chart-panel">
      <p class="chart-title">Current rental cost</p>
      <p class="chart-subtitle">Lambda eight-GPU on-demand USD per PFLOP-second; lower is better</p>
      <div class="chart-frame chart-frame--compact"><canvas id="gpu-rental-cost-chart" role="img" aria-label="Interactive logarithmic chart comparing current cloud rental dollars per PFLOP-second for V100, A100, H100, and B200.">The source data is available in the methodology table below.</canvas></div>
    </section>
  </div>
  <figcaption>Three distinct views share one compute basis. Peak compute uses dense FP16/BF16, except C870's FP32 legacy proxy. Historical buy cost uses release-era standalone or eight-GPU system price divided by GPU count. Current rental cost uses Lambda's eight-GPU on-demand rates accessed July 26, 2026. Lower is better in both cost plots; Rubin has a preliminary compute point but no public cost point.</figcaption>
</figure>

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

### What the three curves measure

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

### A rack-to-die BOM proxy shows where the dollars moved

<p class="key-insight"><strong>Key insight</strong><span>Read accelerator economics from the outside in. The rack first pays for GPUs, memory, communication, CPUs, power, and cooling; only then does it make sense to ask what sits inside one accelerator module and one logic die.</span></p>

<figure class="post-figure post-chart">
  <section class="chart-panel">
    <p class="chart-title">Rubin VR200 NVL72 rack</p>
    <p class="chart-subtitle">Analyst procurement BOM proxy; about USD 7.8M</p>
    <div class="chart-frame"><canvas id="rubin-rack-bom-chart" role="img" aria-label="One hundred percent stacked bar chart showing the estimated Vera Rubin VR200 NVL72 rack cost share across the Rubin GPU line excluding separately accounted memory, memory including HBM4, communication, CPUs, power and cooling, and the remaining platform.">The estimated breakdown is available in the table below.</canvas></div>
  </section>
  <figcaption>The analyst table accounts for GPU and memory separately. Its USD 3.96M Rubin GPU line therefore excludes the separately reported USD 2.00M memory line, which aggregates HBM4, Vera CPU LPDDR5X/SoCAMM, and rack storage. This accounting boundary differs from the physical GPU package, where HBM4 sits beside the logic dies.</figcaption>
</figure>

A 2026 Morgan Stanley estimate puts VR200 NVL72 at USD 7.80 million: USD 3.96 million for the `GPU` line, USD 2.00 million for the separate `Memory` line, USD 720,000 for NVLink Switch and other networking chips, USD 180,000 for Vera CPUs, and the remainder for cooling, power, boards, substrates, passives, assembly, and other platform content.[^rubin-rack-bom] The memory discussion explicitly includes Rubin HBM4 alongside Vera LPDDR5X/SoCAMM and NAND storage, so the GPU line must not be read as a complete HBM-inclusive physical package. The resulting top-level ratio is **GPU line excluding separately accounted memory 50.7%, memory 25.7%, communication 9.2%, CPUs 2.3%, and platform/power/cooling 12.1%**. NVIDIA's public topology confirms why communication has its own bill: an NVL72 domain contains 18 compute trays, nine NVLink Switch trays, and 72 GPUs.[^gb200-rack-topology]

#### Accelerator packages: HBM and packaging dominate

<p class="key-insight"><strong>Key insight</strong><span>Inside an accelerator module, the logic die is not the majority cost. HBM and advanced packaging together account for roughly three quarters of both the H20 and B200 manufacturing proxies.</span></p>

<figure class="post-figure post-chart">
  <div class="chart-grid chart-grid--two">
    <section class="chart-panel">
      <p class="chart-title">H20 accelerator module</p>
      <p class="chart-subtitle">Estimated manufacturing proxy; about USD 2.6K</p>
      <div class="chart-frame"><canvas id="h20-module-bom-chart" role="img" aria-label="One hundred percent stacked bar chart estimating H20 accelerator module manufacturing cost across the logic die, HBM3, CoWoS packaging, and auxiliary module components.">The estimated breakdown is available in the table below.</canvas></div>
    </section>
    <section class="chart-panel">
      <p class="chart-title">B200 accelerator module</p>
      <p class="chart-subtitle">Estimated manufacturing proxy; about USD 6.4K</p>
      <div class="chart-frame"><canvas id="b200-module-bom-chart" role="img" aria-label="One hundred percent stacked bar chart estimating B200 accelerator module manufacturing cost across logic dies, HBM3E, advanced packaging and yield loss, and auxiliary module components.">The estimated breakdown is available in the table below.</canvas></div>
    </section>
  </div>
  <figcaption>These package-level views deliberately keep logic as one line item. H20 is shipment-normalized from Epoch AI component spend; B200 uses Epoch AI's modeled central component estimates. They estimate manufacturing inputs rather than sale price.</figcaption>
</figure>

The H20 proxy divides approximately USD 2.59 billion of Q3-Q4 2024 component spend by roughly one million 2024 shipments, implying about **USD 2,600 per module**, with a broad USD 2,100-3,100 sensitivity range.[^h20-bom-proxy] Its top-level split is **57.9% HBM3, 16.6% CoWoS-S, 15.1% logic, and 10.4% auxiliary components**. This fits the product's role: H20 retains 96 GB of HBM3 and about 4 TB/s of bandwidth while its exported compute configuration is far below H100.[^h20-spec] Its roughly USD 12,000-15,000 2024 sale price additionally reflected margin, software value, channel costs, and market conditions.[^h20-price]

Epoch AI estimates a B200 module at roughly USD 5,700-7,300, centered near **USD 6,400**. Its central inputs are USD 2,900 for 192 GB of physically packaged HBM3E, USD 1,100 for CoWoS-L, USD 900 for two logic dies, USD 1,000 for packaging yield loss, and USD 480 for power delivery, PCB, assembly, and testing.[^b200-bom] The top-level split is therefore **45.5% HBM3E, 32.9% packaging and yield, 14.1% logic, and 7.5% auxiliary components**. Shipping B200 specifications expose 180 GB as usable memory; the model prices the physical 192 GB capacity.

#### Logic dies: a floorplan explains the logic line

<p class="key-insight"><strong>Key insight</strong><span>A logic die is not an ALU slab. Compute regions share silicon with local SRAM and control, while cache, memory controllers, die-to-die links, NVLink, PCIe, and fabric consume substantial visible area.</span></p>

<figure class="post-figure post-chart">
  <div class="chart-grid chart-grid--two">
    <section class="chart-panel">
      <p class="chart-title">Hopper GH100 logic</p>
      <p class="chart-subtitle">Physical-model area share</p>
      <div class="chart-frame"><canvas id="h20-logic-floorplan-chart" role="img" aria-label="One hundred percent stacked bar chart estimating GH100 logic-die area across SM regions, standalone L2, memory and link I/O, and other uncore.">The floorplan shares are available in the table below.</canvas></div>
    </section>
    <section class="chart-panel">
      <p class="chart-title">Blackwell logic</p>
      <p class="chart-subtitle">Pixel-area share of NVIDIA's annotated die image</p>
      <div class="chart-frame"><canvas id="b200-logic-floorplan-chart" role="img" aria-label="One hundred percent stacked bar chart measuring Blackwell logic area across GPC and SM regions, standalone L2, HBM controllers, link and host I/O, and other uncore.">The floorplan shares are available in the table below.</canvas></div>
    </section>
    <section class="chart-panel chart-panel--wide">
      <p class="chart-title">Rubin logic</p>
      <p class="chart-subtitle">Pixel-area share of NVIDIA's annotated die image</p>
      <div class="chart-frame"><canvas id="rubin-logic-floorplan-chart" role="img" aria-label="One hundred percent stacked bar chart measuring Rubin logic area across GPC and SM regions, standalone L2, HBM controllers, link and host I/O, and other uncore.">The floorplan shares are available in the table below.</canvas></div>
    </section>
  </div>
  <figcaption>These are area proxies, not transistor-level teardowns. GH100 uses an open physical power-delivery model. Blackwell and Rubin trace NVIDIA's labeled die-image boundaries with roughly +/-2 percentage-point sensitivity. Every SM or GPC region includes arithmetic, Tensor Cores, registers, local SRAM, schedulers, and control.</figcaption>
</figure>

For GH100, the model assigns approximately **59.9% to complete SM regions, 8.9% to standalone L2, 11.8% to memory and link I/O, and 19.5% to other fabric and uncore**.[^gh100-floorplan] Applying those shares to H20's USD 393 logic line attributes about USD 235, USD 35, USD 46, and USD 77 respectively. Disabled or restricted execution units still occupy physical die area and incur wafer cost.

For Blackwell, tracing NVIDIA's annotated dual-reticle image gives approximately **39.8% GPC/SM regions, 14.3% standalone L2, 11.2% HBM controllers, 20.9% NV-HBI/NVLink/PCIe regions, and 13.8% other uncore**.[^blackwell-floorplan] Applied only to B200's USD 900 logic line, those areas attribute about USD 358, USD 129, USD 101, USD 188, and USD 124. B200 also exposes about 126 MB of shared L2 plus large per-SM register, tensor-memory, and local-memory structures.[^b200-memory]

Rubin's official annotated image gives approximately **29.3% GPC/SM regions, 8.9% standalone L2, 14.2% HBM controllers, 27.3% NV-HBI/NVLink/PCIe regions, and 20.3% other uncore**.[^rubin-floorplan] No disclosed bare-die cost exists. Mechanically applying area to the analyst's USD 55K GPU line would produce USD 16.1K, USD 4.9K, USD 7.8K, USD 15.0K, and USD 11.2K, but those are only attribution values. The analyst accounts for HBM4 elsewhere, while the GPU line may still include logic packaging, test, margin, and commercial value.

<details class="post-details" markdown="1">
<summary>Show the rack, package, and die attribution tables</summary>

| view | attributed category | USD proxy | share | confidence |
| --- | --- | ---: | ---: | --- |
| Rubin VR200 NVL72 rack | Rubin GPU line, excluding separately accounted memory | USD 3,960,000 | 50.7% of rack | analyst procurement estimate; not an HBM-inclusive package cost |
| Rubin VR200 NVL72 rack | memory: HBM4, CPU memory, and storage | USD 2,001,600 | 25.7% of rack | analyst aggregate; not HBM-only |
| Rubin VR200 NVL72 rack | NVLink Switch plus other networking chips | USD 720,000 | 9.2% of rack | sum of two analyst line items |
| Rubin VR200 NVL72 rack | Vera CPUs | USD 180,000 | 2.3% of rack | analyst procurement estimate |
| Rubin VR200 NVL72 rack | power and cooling | USD 148,080 | 1.9% of rack | sum of two analyst line items |
| Rubin VR200 NVL72 rack | boards, substrate, passives, assembly, and other | USD 793,468 | 10.2% of rack | residual from reported total |
| H20 module | logic die | USD 393 | 15.1% of module | low; shipment-normalized logic-cost line |
| H20 module | 96 GB HBM3 | USD 1,505 | 57.9% | medium; shipment-normalized component model |
| H20 module | CoWoS-S packaging | USD 432 | 16.6% | medium; shipment-normalized component model |
| H20 module | module auxiliary components | USD 269 | 10.4% | medium; shipment-normalized component model |
| B200 module | two logic dies | USD 900 | 14.1% of module | modeled component estimate |
| B200 module | 192 GB physical HBM3E; 180 GB exposed | USD 2,900 | 45.5% | modeled component estimate |
| B200 module | CoWoS-L plus package yield loss | USD 2,100 | 32.9% | modeled component estimate |
| B200 module | module auxiliary components | USD 480 | 7.5% | modeled component estimate |
| H20 logic line | complete SM regions | USD 235 | 59.9% of logic; 9.1% of module | low; modeled bounding regions, not ALU-only |
| H20 logic line | standalone L2 region | USD 35 | 8.9% of logic; 1.34% of module | low; modeled 72 mm2 bounding region |
| H20 logic line | memory-controller plus NVLink / PCIe I/O regions | USD 46 | 11.8% of logic; 1.78% of module | low; modeled dimensions, with 12-controller scaling |
| H20 logic line | other fabric and uncore | USD 77 | 19.5% of logic; 2.94% of module | low; residual die area |
| B200 logic line | complete GPC / SM regions | USD 358 | 39.8% of logic; 5.6% of module | low-medium; diagram-measured area |
| B200 logic line | standalone L2 regions | USD 129 | 14.3% of logic; 2.0% of module | low-medium; diagram-measured area |
| B200 logic line | HBM controller regions | USD 101 | 11.2% of logic; 1.6% of module | low-medium; diagram-measured area |
| B200 logic line | NV-HBI, NVLink, and host-I/O regions | USD 188 | 20.9% of logic; 3.0% of module | low-medium; diagram-measured area |
| B200 logic line | other fabric and uncore | USD 124 | 13.8% of logic; 1.9% of module | low; residual diagram area |
| Rubin GPU line attribution | complete GPC / SM regions | USD 16.1K per GPU | 29.3% of GPU line | low; excludes separately accounted memory; not cost |
| Rubin GPU line attribution | standalone L2 regions | USD 4.9K per GPU | 8.9% of GPU line | low; excludes separately accounted memory; not cost |
| Rubin GPU line attribution | HBM controller regions | USD 7.8K per GPU | 14.2% of GPU line | low; excludes separately accounted memory; not cost |
| Rubin GPU line attribution | NV-HBI, NVLink, and host-I/O regions | USD 15.0K per GPU | 27.3% of GPU line | low; excludes separately accounted memory; not cost |
| Rubin GPU line attribution | other fabric and uncore | USD 11.2K per GPU | 20.3% of GPU line | low; residual area attribution; not cost |

Read the rows at their stated level. Rack rows are procurement estimates, module rows are manufacturing proxies, and die rows are area attributions within a disclosed logic-cost line or the analyst's memory-exclusive Rubin GPU line. They are neither NVIDIA's internal costs nor retail margins. Diagram tracing measures labeled bounding regions, not transistor occupancy, and cannot separate SRAM from arithmetic inside an SM. TechInsights sells measured GH100 and B200 floorplan analyses, but the block-utilization tables are not public.[^gh100-floorplan][^blackwell-floorplan]

</details>

### Die logic cost: arithmetic density versus wafer price

<p class="key-insight"><strong>Key insight</strong><span>FP16 arithmetic occupies far less silicon than it did at 45nm, but rising leading-edge wafer prices make dollars per ALU fall more slowly than area per ALU.</span></p>

<figure class="post-figure post-chart">
  <div class="chart-grid chart-grid--two">
    <section class="chart-panel">
      <p class="chart-title">Estimated raw FP16 ALU cost</p>
      <p class="chart-subtitle">USD per one million multiply-plus-add datapaths; logarithmic scale</p>
      <div class="chart-frame chart-frame--compact"><canvas id="alu-cost-chart" role="img" aria-label="Interactive logarithmic chart showing the estimated raw cost of one million FP16 multiply-plus-add datapaths from 45 nanometer through N2 and Intel 18A.">The process-node estimates are available below.</canvas></div>
    </section>
    <section class="chart-panel">
      <p class="chart-title">Advanced wafer price</p>
      <p class="chart-subtitle">Foundry sale price per 300 mm wafer; logarithmic scale</p>
      <div class="chart-frame chart-frame--compact"><canvas id="wafer-price-chart" role="img" aria-label="Interactive logarithmic chart showing approximate advanced wafer prices across 28 nanometer through 3 nanometer.">The methodology is available below.</canvas></div>
    </section>
  </div>
  <figcaption>The same node shrink that fits more FP16 datapaths also raises the price of the wafer carrying them. The left chart combines both effects as a lower-bound manufacturing proxy, not a tensor-core or finished-GPU cost.</figcaption>
</figure>

The physical hierarchy starts inside the logic die. A normalized FP16 multiply-plus-add datapath shrinks from about 3,000 um2 at 45nm to roughly 206 um2 at 7nm and 87 um2 at 3nm. Lower precision and matrix engines can push useful compute density further, but only when software and the memory system keep those lanes fed.[^horowitz][^alu-area-cost]

<details class="post-details" markdown="1">
<summary>Show the logic-cost model and process anchors</summary>

The chart uses one deliberately narrow proxy:

<div class="math-block">
$$
\text{raw ALU cost}
\approx
\text{ALU area}
\times
\text{wafer price per mm}^2
$$
</div>

The 45nm baseline adds Horowitz's published 16-bit FP add and multiply areas. Later rows scale that same logical datapath with public logic-density estimates and public wafer-price anchors.[^horowitz][^cmos-cost][^logic-density-28-7][^process-density-5nm][^process-density-3nm][^next-node-density][^cset-wafer-cost][^wafer-pricing][^n2-wafer-price]

| node | era | estimated FP16 mul+add area | raw cost per 1M units |
| --- | ---: | ---: | ---: |
| 45nm | 2007 | 3,000 um2 | USD 85 |
| 28nm | 2010 | 1,225 um2 | USD 52 |
| 16nm | 2015 | 649 um2 | USD 37 |
| 7nm | 2018 | 206 um2 | USD 27 |
| 5nm | 2020 | 136 um2 | USD 33 |
| 3nm | 2024 | 87 um2 | USD 24 |
| N2 estimate | 2026 | 60 um2 | USD 25 |
| Intel 18A estimate | 2026 | 79 um2 | USD 33 |

SMIC N+1 through N+3 form a separate 7nm-class branch: teardowns show continued density improvement, but public wafer prices are insufficient for a comparable dollar estimate.[^smic-n1-n2][^smic-n3] Huawei LogicFolding is also excluded because it increases package-footprint density through active-die stacking rather than defining a directly comparable planar node.[^huawei-logic-folding]

This proxy omits yield, routing, registers, control, clocking, SRAM, verification, and packaging. It should therefore be read as the direction of raw logic economics, not as an ALU allocation within a commercial GPU.

</details>

## 2. Memory cost: on-die SRAM to package HBM

<p class="key-insight"><strong>Key insight</strong><span>Moving outward from the die buys capacity, but costs latency and energy; HBM buys back bandwidth at a high package-level price.</span></p>

<figure class="post-figure post-chart">
  <div class="chart-grid chart-grid--two">
    <section class="chart-panel">
      <p class="chart-title">On-die SRAM cost proxy</p>
      <p class="chart-subtitle">Raw bitcell USD/MB lower bound</p>
      <div class="chart-frame chart-frame--compact"><canvas id="sram-cost-chart" role="img" aria-label="Interactive chart showing the raw SRAM bitcell cost proxy across 28 nanometer, 7 nanometer, 5 nanometer, and 3 nanometer or N2-class nodes.">The methodology is available below.</canvas></div>
    </section>
    <section class="chart-panel">
      <p class="chart-title">Package HBM capacity price</p>
      <p class="chart-subtitle">Modeled USD/GB; HBM4 is projected</p>
      <div class="chart-frame chart-frame--compact"><canvas id="hbm-capacity-cost-chart" role="img" aria-label="Interactive chart showing modeled HBM dollars per gigabyte for HBM2e, HBM3, HBM3e, and projected HBM4.">The methodology is available below.</canvas></div>
    </section>
    <section class="chart-panel chart-panel--wide">
      <p class="chart-title">Package HBM bandwidth price</p>
      <p class="chart-subtitle">Modeled USD per TB/s</p>
      <div class="chart-frame chart-frame--compact"><canvas id="hbm-bandwidth-cost-chart" role="img" aria-label="Interactive chart showing modeled HBM dollars per terabyte per second for HBM2e through projected HBM4.">The methodology is available below.</canvas></div>
    </section>
  </div>
  <figcaption>The hierarchy has two different price units: on-die SRAM is constrained by silicon area per MB, while package HBM is purchased for both GB of capacity and TB/s of bandwidth.</figcaption>
</figure>

On-die SRAM keeps reused data closest to the arithmetic, but each additional MB consumes expensive logic area. HBM holds far more model state and KV cache, yet it adds stacked memory, interposers, controllers, and package complexity. AI infrastructure bridges the two with tiling, fusion, quantization, paging, and cache placement.

<details class="post-details" markdown="1">
<summary>Show the SRAM and HBM methodology</summary>

### On-die SRAM cost

Because SRAM has no public spot price, the chart uses a lower bound:

<div class="math-block">
$$
\text{raw SRAM cost per MB}
\approx
\text{bitcell area per MB}
\times
\text{wafer price per mm}^2
$$
</div>

| node | SRAM bitcell | wafer price used | raw SRAM cost proxy |
| --- | ---: | ---: | ---: |
| 28nm | 0.127 um2 | USD 3,000 | USD 0.045/MB |
| 7nm | 0.027 um2 | USD 9,346 | USD 0.030/MB |
| 5nm | 0.021 um2 | USD 16,988 | USD 0.042/MB |
| 3nm / 2nm-class | 0.021 um2 | USD 19,500 | USD 0.049/MB |

The proxy excludes periphery, redundancy, cache tags, routing, and yield. It nevertheless captures the post-7nm tension: SRAM cells shrink slowly while wafer prices rise.[^tsmc-28nm-sram][^tsmc-5nm-sram][^tsmc-2nm-sram][^cset-wafer-cost][^wafer-pricing] GPU designers respond by spending more die area on reuse: L2 grew from about 4 MB on P100 to 40 MB on A100, 50 MB on H100, and roughly 126 MB in public B200 analysis.[^a100][^h100][^chips-b200-cache]

### Package HBM cost

HBM is expensive capacity purchased for proximity and bandwidth. Modeled prices rise from about USD 6/GB for HBM2e to USD 18/GB at the HBM3e peak, with projected HBM4 around USD 16.5/GB. The corresponding bandwidth proxy ranges from about USD 209 to USD 352 per TB/s.[^stanford-memory-prices][^rambus-hbm]

| GPU | HBM | capacity | bandwidth |
| --- | --- | ---: | ---: |
| P100 | HBM2 | 16 GB | 720 GB/s |
| A100 | HBM2 / HBM2e | 40-80 GB | 1,555-2,039 GB/s |
| H100 | HBM3 | 80 GB | over 3 TB/s |
| B200 | HBM3e | 180 GB | 7.7 TB/s |
| Rubin | HBM4 | 288 GB | 22 TB/s |

At NVL72 scale, Rubin aggregates 20.7 TB of HBM4 capacity and 1,580 TB/s of memory bandwidth across 72 GPUs.[^p100][^a100][^h100][^b200-lenovo][^vera-rubin-spec]

</details>

## 3. Communication cost: package to rack to cluster

<p class="key-insight"><strong>Key insight</strong><span>Every step away from the die expands the compute domain, but adds more latency, contention, topology, and equipment cost.</span></p>

<figure class="post-figure">
  <img src="{{ '/assets/scale-up-scale-out-excalidraw.svg' | relative_url }}" alt="Excalidraw topology showing package links connecting GPU dies, NVLink and NVSwitch connecting GPUs within rack-scale domains, and Ethernet or InfiniBand connecting racks into a cluster.">
  <figcaption>Read the hierarchy from near to far: package links join dies, scale-up links make a box or rack behave like one machine, and scale-out fabric connects many rack-scale machines.</figcaption>
</figure>

| physical level | representative link | cost and performance unit |
| --- | --- | --- |
| package | NV-HBI and die-to-die fabric | die area, package complexity, bytes/s per edge |
| rack scale-up | NVLink and NVSwitch | bandwidth per GPU or rack-scale domain |
| cluster scale-out | Ethernet and InfiniBand | ports, optics, NICs, cables, and USD/Gb/s |

The software-visible communication cost can be approximated with one latency-bandwidth model:

<div class="math-block">
$$
T_{\text{comm}}
\approx
\alpha n_{\text{messages}}
+
\frac{\text{bytes moved}}{B_{\text{effective}}}
$$
</div>

Here `alpha` is latency per message and `B_effective` is achieved bandwidth after topology, protocol, contention, and collective overhead. Tensor parallelism, MoE routing, and disaggregated serving increase either the message count, the bytes moved, or both. NVIDIA's scale-up bandwidth per GPU rises from 900 GB/s on Hopper to 1.8 TB/s on Blackwell and 3.6 TB/s on Rubin; NVL72 aggregate bandwidth rises from 130 TB/s to 260 TB/s between Blackwell and Rubin.[^nvlink][^hgx-rubin]

### Cluster fabric cost: dollars per bandwidth

<p class="key-insight"><strong>Key insight</strong><span>Switch bandwidth became much cheaper, but AI systems spend the gain on larger communication domains and more distributed inference.</span></p>

<figure class="post-figure post-chart">
  <div class="chart-grid chart-grid--two">
    <section class="chart-panel">
      <p class="chart-title">Scale-out port speed</p>
      <p class="chart-subtitle">Front-panel Gb/s; logarithmic scale</p>
      <div class="chart-frame chart-frame--compact"><canvas id="interconnect-speed-chart" role="img" aria-label="Interactive logarithmic chart showing scale-out port speed increasing from 10 gigabits per second in 2008 to 800 gigabits per second in 2026.">The switch data is available below.</canvas></div>
    </section>
    <section class="chart-panel">
      <p class="chart-title">Switch cost per bandwidth</p>
      <p class="chart-subtitle">Chassis-only USD/Gb/s proxy; logarithmic scale</p>
      <div class="chart-frame chart-frame--compact"><canvas id="interconnect-cost-chart" role="img" aria-label="Interactive logarithmic chart showing switch dollars per gigabit per second falling from about 72 dollars to about one dollar.">The switch data is available below.</canvas></div>
    </section>
  </div>
  <figcaption>Port speed rose from tens to hundreds of Gb/s while chassis-only cost fell from about USD 72 to USD 1 per Gb/s. This proxy excludes optics, NICs, cables, support, power, and topology overhead.</figcaption>
</figure>

InfiniBand progressed from QDR through EDR, HDR, NDR, and XDR; Ethernet moved from 40/100GbE to 200/400GbE and then 800GbE.[^infiniband-rates][^ethernet-100g][^ethernet-400g][^ethernet-800g] Faster links do not automatically lower the cluster network bill: larger collective domains, expert routing, and KV-cache movement consume the added bandwidth.

<details class="post-details" markdown="1">
<summary>Show communication workloads and switch-price anchors</summary>

| workload pattern | dominant communication |
| --- | --- |
| tensor parallel | activation all-reduce and all-gather |
| pipeline parallel | boundary activations and pipeline bubbles |
| expert parallel / MoE | token dispatch and all-to-all routing |
| disaggregated serving | KV-cache movement and prefill/decode handoff |

The scale-out price proxy divides public switch chassis price by front-panel bandwidth. It measures the switching layer, not a complete network.[^nexus-price][^mellanox-switch-prices][^sn5610-price][^sn5610-specs]

| switch proxy | era | ports x speed | price anchor | USD/Gb/s |
| --- | ---: | ---: | ---: | ---: |
| Cisco Nexus 5020 | 2008 | 40 x 10 Gb/s | USD 28,770 | USD 72 |
| Mellanox SB7800 EDR | 2015 | 36 x 100 Gb/s | USD 10,259 | USD 2.85 |
| Mellanox QM8700 HDR | 2018 | 40 x 200 Gb/s | USD 18,740 | USD 2.34 |
| NVIDIA QM9700 NDR | 2022 | 64 x 400 Gb/s | USD 32,870 | USD 1.28 |
| NVIDIA SN5610 800GbE | 2026 | 64 x 800 Gb/s | USD 51,999 | USD 1.02 |

NVIDIA's Quantum-X800 documentation lists 72-port and 144-port XDR systems at 800 Gb/s per port, reaching 115.2 Tb/s for the 4U system.[^quantum-x800]

</details>

Across all three levels, infrastructure creates value by moving fewer bytes, keeping reusable state near compute, overlapping communication with kernels, and choosing a parallelism plan that converts costly bandwidth into useful tokens.

## Edge becomes a separate placement frontier

<p class="key-insight"><strong>Key insight</strong><span>Edge systems must optimize useful local intelligence within a product-level memory, bandwidth, thermal, power, and price envelope.</span></p>

Edge hardware changes the accounting unit. A phone, AI PC, or robotics box is bounded by shared memory, bandwidth, thermals, battery life, and a product-level price rather than by rack power and HBM alone. Peak TOPS is useful, but it does not tell us how large a model fits, how quickly its weights can be streamed during decoding, or how much benchmark quality survives quantization.

[Part 4 follows this edge intelligence envelope directly](/2026/07/03/ai-infra-edge-intelligence/): how mobile, AI PC, and AI-box hardware evolved; how their product-level dollars per advertised performance changed; how edge-fit language models improved; and what those trends imply for local agents.

## Prediction: heterogeneity becomes the default

<p class="key-insight"><strong>Key insight</strong><span>The future AI system is a coordinated heterogeneous package whose software places each workload on the right compute, memory, and communication resource.</span></p>

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
[^h20-bom-proxy]: Epoch AI, [AI Chip Components dataset and methodology](https://epoch.ai/data/ai-chip-components-documentation), accessed 2026-08-09. The median H20 rows for Q3 and Q4 2024 sum to approximately USD 392 million logic, USD 431 million CoWoS, USD 1.500 billion HBM, and USD 268 million auxiliary spend. Epoch AI's [AI Chip Sales methodology](https://epoch.ai/data/ai-chip-sales-documentation/methodology) cites approximately one million H20 shipments in 2024. Dividing component spend by shipments gives the article's USD 2.6K module proxy; the production and shipment periods are not perfectly matched.
[^h20-spec]: NVIDIA documentation identifies the [H20 SXM5 as a 96 GB Hopper GPU](https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/reference/hopper.html). [Contemporary H20 product reporting](https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidia-to-make-12-billion-selling-ai-gpus-to-china) lists 96 GB of HBM3, 4.0 TB/s of memory bandwidth, and 296 FP8 TFLOP/s.
[^gh100-floorplan]: NVIDIA's [Hopper architecture description](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) specifies an 814 mm2 full GH100 die with 144 SMs, 60 MB L2, 12 memory controllers, NVLink, and PCIe. Anasim's open [H100 full-chip power-delivery model](https://www.anasim.com/articles/pdn-resonance-h100-analysis) represents 144 SM regions at 1.84 x 1.84 mm each, two L2 regions at 12 x 3 mm each, ten enabled memory-controller regions at 1 x 3.5 mm each, NVLink at 27.5 x 1.2 mm, and PCIe at 21 x 1 mm. This article scales the controller region from ten to the full design's twelve. The resulting allocation is approximately 59.9%, 8.8%, 11.8%, and 19.5% of die area for SM regions, L2, named I/O, and residual uncore. These are engineering-model bounding regions, not measured transistor utilization. TechInsights confirms that its paid [GH100 digital floorplan analysis](https://www.techinsights.com/zh-cn/node/51295) contains measured functional-block sizes and die-utilization percentages, but does not publish the table on the accessible page.
[^h20-price]: Yelin Mo and Brenda Goh, Reuters, ["Nvidia's new China-focused AI chip set to be sold at similar price to Huawei product"](https://m.uk.investing.com/news/stock-market-news/exclusivenvidias-new-chinafocused-ai-chip-set-to-be-sold-at-similar-price-to-huawei-product-3319402), 2024. NVIDIA distributor pricing was reported at USD 12,000-15,000 per card; later pricing varied with demand and export restrictions.
[^b200-bom]: Venkat Somala, Epoch AI, ["NVIDIA's B200 costs around USD 6,400 to produce, with memory accounting for half"](https://epoch.ai/data-insights/b200-cost-breakdown), 2025. The model uses public reporting, analyst estimates, company disclosures, and Monte Carlo ranges; it estimates variable manufacturing cost rather than server price or NVIDIA's full cost structure.
[^b200-memory]: Cornell Virtual Workshop, ["GPU Memory Levels"](https://cvw.cac.cornell.edu/gpu-architecture/gpu-memory/memory_levels), accessed 2026-08-09, summarizing B200's 126 MB L2, 256 KB register file per SM, and 256 KB unified L1/shared memory per SM. The article's cost allocation is an explicit proxy, not Cornell's estimate.
[^blackwell-floorplan]: NVIDIA's [annotated Blackwell Ultra dual-reticle image and architecture description](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra/) show eight GPCs, distributed L2, HBM controllers, NV-HBI, NVLink, NVLink-C2C, and PCIe. Cornell's [Blackwell chip diagram notes](https://cvw.cac.cornell.edu/gpu-architecture/horizon-gpus-blackwell-b200/blackwell_chip) state that the B200 and B300 layouts are practically identical. The percentages here trace the labeled white boundaries against the complete dual-die rectangle; moving ambiguous boundaries by several pixels changes major categories by roughly +/-2 percentage points. TechInsights confirms a physical [B200 processor floorplan analysis](https://www.techinsights.com/blog/nvidia-blackwell-b200-processor-floorplan-analysis), but its detailed block table is not public.
[^rubin-floorplan]: NVIDIA's July 2026 [Rubin GPU architecture article](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/) identifies two reticle-limited dies, 336 billion transistors, 224 SMs, 896 Tensor Cores, 288 GB HBM4, and the labeled GPC, L2, HBM-controller, NV-HBI, NVLink, NVLink-C2C, and PCIe regions used here. The percentages trace the white boundaries in NVIDIA's Figure 2 against the complete two-die rectangle, with roughly +/-2 percentage-point boundary sensitivity. They are image-area measurements, not a teardown or transistor-level floorplan.
[^rubin-rack-bom]: Anton Shilov, Tom's Hardware, ["Nvidia's memory costs soar 485%, latest AI systems now cost USD 7.8 million to build"](https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidias-memory-costs-soar-485-percent-latest-ai-systems-now-cost-usd7-8-million-to-build-memory-now-comprises-25-percent-of-the-total-cost-rubin-gpus-a-mere-usd50-000-apiece), 2026; [full line-item transcription of the circulated Morgan Stanley table](https://log.eurekapu.com/vr200-nvl72-bom-memory-cost/), 2026. The source table lists `GPU` and `Memory` separately, while the accompanying analysis attributes the memory line to HBM4, Vera LPDDR5X/SoCAMM, and NAND. This is a forward-looking analyst procurement estimate, not a public NVIDIA BOM or independently verified teardown.
[^gb200-rack-topology]: NVIDIA, ["Understanding Your Grace-Blackwell Systems"](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/system.html), documenting the NVL72 reference configuration with 18 compute trays, nine NVLink Switch trays, and 72 GPUs. Rubin pricing is not inferred from this source; it is used only as the rack-topology anchor.
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
<script defer src="{{ '/assets/series-charts.js' | relative_url }}?v=20260810e"></script>
