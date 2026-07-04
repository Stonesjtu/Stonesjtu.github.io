---
layout: post
title: "[WIP] AI infra and tokenomics"
topic: "AI infrastructure"
sequence: 8
source_url: https://app.notion.com/p/38d2ec4bb1f0808ea061d11de43d93a6
source_label: "Original outline on Notion"
excerpt: "Why AI infrastructure has become the scarce layer of the modern stack, and why token economics increasingly depends on model, systems, and chip co-design."
---

This post is about a simple systems question:

```text
How many useful tokens can a stack produce per dollar, watt, second, and engineer-hour?
```

For LLM systems, the token is the unit where model quality, hardware capacity, serving policy, and product economics meet. Training consumes tokens to create capability. Inference consumes tokens to deliver capability. Agentic workflows consume even more tokens to plan, call tools, verify work, and recover from failure. Token cost is therefore not just a pricing metric. It is a full-stack performance metric.

The core argument is:

- Model demand is still scaling: parameters, training tokens, context length, multimodal inputs, and inference-time compute all keep increasing.
- Hardware supply is improving, but less like the old free lunch: power, memory bandwidth, packaging, and manufacturing cost are now first-order constraints.
- The gap is closed by infrastructure: model architecture, serving systems, kernels, compilers, accelerator backends, memory hierarchy, networking, and chip design have to be optimized together.

This is why AI infrastructure is not a support layer around the model. It is part of the model's unit economics. OpenAI's 2018 analysis found that compute used in the largest AI training runs had been doubling every 3.4 months from 2012 onward.[^openai-compute] Epoch AI's later study gives a more conservative but still aggressive estimate: since the early deep learning era, training compute for milestone ML systems has doubled roughly every 5 to 6 months.[^epoch-compute] That demand curve is faster than what hardware economics can hide automatically.

<figure class="post-figure">
  <img src="{{ '/assets/ai-infra-demand-supply.svg' | relative_url }}" alt="Conceptual chart showing AI model and token demand rising faster than accelerator compute, memory bandwidth, and power budgets.">
  <figcaption>AI infrastructure matters because the curve to serve is steeper than the curve hardware gives us for free.</figcaption>
</figure>

From conventional infra to agentic infra
---

<figure class="post-figure">
  <img src="{{ '/assets/infra-evolution-agentic.svg' | relative_url }}" alt="Timeline diagram showing infrastructure evolving from conventional request serving to machine learning jobs, deep learning training steps, LLM tokens, and agentic task loops.">
  <figcaption>The important shift is the unit of work: infrastructure moves from operating services to scheduling accelerator-heavy token loops that must complete useful tasks.</figcaption>
</figure>

The useful way to compare infrastructure generations is to ask what the scheduler is really scheduling.

| era | unit of work | scarce resource | infra objective |
| --- | --- | --- | --- |
| conventional infra | request / transaction | CPU, storage, availability | keep services reliable |
| ML infra | experiment / tensor job | GPU hours, data pipelines | make iteration reproducible |
| deep learning infra | training step / kernel graph | accelerator utilization | keep devices fed |
| LLM infra | token / KV cache block | memory bandwidth, batch slots | optimize prefill/decode economics |
| agentic infra | task loop / tool call | total tokens, state, retries | complete useful work cheaply |

Conventional infrastructure mostly optimizes request handling: deployment, isolation, storage, observability, and failure recovery. ML infrastructure moves the unit of work to tensor programs and experiments, so GPU allocation, data loading, checkpoints, and artifact tracking become central.

Deep learning infrastructure makes the accelerator the main computer. The system is only efficient when the input pipeline, graph executor, collective communication, and kernels keep devices busy. LLM infrastructure adds a serving-specific memory problem: KV cache placement, batching, prefill/decode separation, quantization, routing, and latency SLOs.

Agentic infrastructure adds feedback loops. A request can expand into planning, retrieval, tool calls, code execution, test runs, retries, and verification. The performance target is no longer "tokens per second" alone. It is useful completed work per token.

Why infra became so resource-demanding
---

Two scaling modes show why the infra problem does not stay fixed.

<figure class="post-figure">
  <img src="{{ '/assets/strong-weak-scaling.svg' | relative_url }}" alt="Two-panel diagram comparing strong scaling with fixed per-request work and weak scaling with growing LLM per-request work.">
  <figcaption>Pre-LLM infrastructure mostly optimizes fixed per-user work. LLM infrastructure optimizes a workload that grows with model size, token length, and inference-time steps.</figcaption>
</figure>

In strong scaling, the workload is fixed and more hardware is added to finish sooner. The limiting terms are communication, synchronization, stragglers, and idle time. Adding more devices eventually exposes imperfect overlap and coordination overhead.

Pre-LLM consumer infrastructure mostly fits this pattern. A video stream, social-network feed, or game session can be expensive at global scale, but the compute per user is usually bounded by the product workload and paced by active user actions:

<div class="math-block">
$$
\begin{aligned}
W_{\text{pre}} &\approx O(1) \\
R_{\text{pre}} &\approx R_{\text{user}}
\end{aligned}
$$
</div>

Here \(W\) is work per request and \(R\) is requests per user. In the pre-LLM loop, the user usually stays in the loop: click, watch, scroll, type, play, wait. That bounds both the work per request and how many expensive requests one person can trigger.

For a fixed workload $W$, strong scaling tries to reduce time by adding parallel resources $P$:

<div class="math-block">
$$
T_{\text{strong}}(W, P) \approx \frac{W}{P} + T_{\text{comm}}(P)
$$
</div>

In other words, once the product behavior is fixed, infra optimization mostly reduces the constant factor for serving the same request, stream, feed, or frame. The user count grows, but the compute size of each user's unit of work does not keep expanding because the model got larger or the context got longer.

LLM infrastructure is different. Both terms can grow. Let \(M\) be model size, \(T\) be total tokens, \(S\) be inference-time steps such as tool calls, retries, or verification passes, \(A\) be concurrent agents per user, and \(R_{\text{agent}}\) be the request rate inside each agent loop:

<div class="math-block">
$$
\begin{aligned}
W_{\text{post}} &\approx O(M \cdot T \cdot S) \\
R_{\text{post}} &\approx R_{\text{user}} + A \cdot R_{\text{agent}}
\end{aligned}
$$
</div>

This intentionally drops a separate attention/cache term. KV cache size, attention pattern, and memory bandwidth still matter a lot in real systems, but they are implementation constraints for this high-level model. The important point is simpler: post-LLM serving has two weak-scaling axes. Work per request grows with \(M\), \(T\), and \(S\); request count per user can grow because a user can define a goal once and let agents run for hours, call tools, inspect files, run tests, monitor state, and spawn concurrent subtasks.

This is why LLMs fit weak scaling better. In weak scaling, the hardware grows and the workload grows with it. More compute is used to train larger models, consume more data, extend context length, add modalities, and spend more compute at inference time. The bottleneck moves instead of disappearing:

- More GPUs make interconnect topology and collective overlap important.
- Longer context turns KV cache capacity and bandwidth into serving constraints.
- Larger batches improve throughput but change latency and memory pressure.
- MoE and routing improve parameter efficiency but add load-balancing problems.
- Test-time compute improves quality but increases token budget variance.

The financial implication is the important part. If an infra optimization reduces unit cost by a factor $r$, then:

<div class="math-block">
$$
\begin{aligned}
\text{cost}_{\text{base}} &= N \cdot R \cdot W \cdot c \\
\text{cost}_{\text{opt}} &= N \cdot R \cdot W \cdot c \cdot r \\
\text{savings} &= N \cdot R \cdot W \cdot c \cdot (1-r)
\end{aligned}
$$
</div>

Under strong scaling, \(W\approx O(1)\) and \(R\) is mostly user-paced, so savings mostly track traffic volume \(N\). Under weak scaling, \(W\) grows with model size, context length, generated tokens, and agent steps, while \(R\) can grow with background agents and concurrent subtasks. The same 20% kernel, compiler, cache, or batching win is therefore magnified twice: by more work per request and by more requests per user goal. This is the "the more you buy, the more you save" property of AI infra: as token demand scales up, every percentage point of efficiency converts into a larger absolute dollar saving.

The scaling-law literature explains why the pressure is not arbitrary. Kaplan et al. found language-model loss following power laws with model size, dataset size, and training compute across many orders of magnitude.[^kaplan] Hoffmann et al. showed that compute-optimal training needs model size and training tokens to scale together; Chinchilla improved quality under the same compute budget by training a smaller model on more tokens.[^chinchilla]

The engineering takeaway is narrower than "make models bigger." Parameters, data, training FLOPs, context, and inference FLOPs are coupled. Infrastructure determines where that compute can be spent efficiently.

The AI infra ecosystem
---

No single layer owns token cost. Each layer chooses constraints for the next one:

- Algorithm: objective, data mixture, verification signal, test-time strategy.
- Model: dense vs sparse, attention pattern, context length, activation shape, precision tolerance.
- Serving: batching, admission control, routing, KV cache policy, speculative decoding, SLO handling.
- Framework: graph capture, dynamic shape support, memory planning, distributed APIs.
- Compiler and runtime: fusion, tiling, scheduling, communication overlap, layout selection.
- Kernel: attention, GEMM, normalization, sampling, collectives, cache movement.
- Accelerator backend: datatypes, tensor core access, streams, DMA, topology awareness.
- Hardware: compute, SRAM, HBM, interconnect, packaging, power, manufacturing yield.

A locally good choice can be globally bad. A model architecture can look clean but fragment KV cache. A compiler pass can improve a microbenchmark but increase memory pressure. A serving policy can raise throughput while violating latency SLOs. A chip feature can be impressive but unreachable from the framework.

Most high-leverage infra work is therefore cross-layer work.

<figure class="post-figure">
  <img src="{{ '/assets/ai-infra-token-stack.svg' | relative_url }}" alt="Stack diagram showing algorithm, model, serving, framework, compiler, kernel, accelerator, memory, network, chip design, and manufacturing all contributing to token economics.">
  <figcaption>The cost and latency of one useful token is the sum of many cross-layer decisions.</figcaption>
</figure>

Infra engineering is the exciting layer
---

The reason infra engineering is high-leverage is that it sits at the constraint boundary:

```text
model demand grows faster than cheap hardware supply
```

Single-thread performance stopped being a free source of software speedup a long time ago.[^sutter] Dark silicon made the power limit explicit: a chip can have many transistors that cannot all be active under the power envelope.[^dark-silicon] For AI workloads, data movement is often a larger constraint than arithmetic. Horowitz's energy numbers are a useful mental model: moving data from memory can cost orders of magnitude more energy than simple arithmetic.[^horowitz]

Modern accelerators respond with HBM, larger on-chip memories, tensor cores, lower-precision formats, faster interconnects, and tighter packaging. But those features only translate into token throughput if the software exposes locality, regularity, and parallelism.

So the practical loop is:

1. Measure the real workload: prefill/decode ratio, cache residency, batch distribution, communication time, kernel hotspots.
2. Change the model or serving policy when the bottleneck is structural.
3. Change kernels, compiler lowering, layouts, and runtime scheduling when the bottleneck is execution.
4. Feed the remaining constraints back into hardware and model design.

Tokenomics is the visible metric for that loop. The question is not just whether a model is intelligent. It is how much useful intelligence the system can deliver per dollar, watt, second, and engineer-hour.

Models are still scaling
---

Scaling has multiple knobs:

- Model scale: parameters, activation size, depth, sparsity, experts.
- Training scale: tokens, data quality, optimizer budget, parallelism efficiency.
- Context scale: KV cache size, attention pattern, retrieval strategy.
- Inference scale: samples, reasoning steps, tool calls, verification passes.

The last knob is increasingly important. Brown et al. studied repeated sampling and found that solution coverage can continue improving as sample count grows over several orders of magnitude, with automatic verification converting samples into better measured performance in domains such as code and formal proofs.[^large-language-monkeys]

This changes the serving objective. A system may improve quality by spending more inference compute, but the product only wins if the extra compute is controlled. Good infra therefore does two things:

1. Lower the cost of each token.
2. Reduce the number of tokens required to complete the task.

Chips are slowing down
---

Hardware is still improving quickly, but the improvement is more specialized and more conditional.

The constraints in this section have a hierarchy:

| layer | question | useful price metric |
| --- | --- | --- |
| compute | How much math can one accelerator expose? | FP16-normalized compute per GPU-hour |
| ALU manufacturing | How cheaply can arithmetic lanes be replicated? | raw ALU area-cost proxy |
| on-chip SRAM | How many hot bytes can stay near compute? | raw SRAM area-cost proxy per MB |
| off-chip DRAM / HBM | How much capacity and bandwidth can the package feed? | USD/GB and USD per TB/s |
| manufacturing | How expensive is each new square millimeter of silicon? | USD per 300 mm wafer |
| interconnect | How expensive is coordination across accelerators? | communication time per collective / all-to-all |

### 1. Compute: peak math is now conditional

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

Blackwell continues the shift. NVIDIA's DGX B200 system lists 144 PFLOP/s FP4 Tensor Core performance across eight Blackwell GPUs, roughly 18 PFLOP/s per GPU at the published system level, while Blackwell Ultra emphasizes 15 PFLOP/s dense NVFP4 per GPU.[^blackwell-b200][^blackwell-ultra] NVIDIA's Rubin announcement lists 50 PFLOP/s NVFP4 inference compute per Rubin GPU, and Vera Rubin NVL144 CPX is framed around 8 exaFLOP/s of rack-scale AI performance for massive-context inference.[^rubin][^rubin-cpx]

These figures are not an apples-to-apples speedup curve. The datatype, sparsity mode, memory system, and programming model all changed. That is the important part. GPU progress came from changing the numerical contract: CUDA, SIMT execution, HBM, NVLink, tensor cores, TF32, BF16, FP8, FP4, sparsity, and compiler/runtime support made model structure visible to hardware.

### 2. Compute per dollar: normalize before comparing

To make the price-performance curve concrete, use a simple rental-equivalent metric:

<div class="math-block">
$$
\Pi = \frac{\text{peak TFLOP/s}}{\text{USD per GPU-hour}},
\qquad
\text{PFLOP-s per USD} = 3.6\Pi
$$
</div>

One TFLOP/s sustained for one hour produces 3.6 PFLOP-s. To keep the comparison closer to apples-to-apples, the table uses dense FP16/BF16 Tensor Core throughput where the GPU supports it. For Tesla C870, which predates FP16 Tensor Cores, I use FP32 peak as a legacy proxy. For H100 and B200, I use dense FP16/BF16, not FP8/FP4 and not sparse-mode peaks.[^h100-spec][^b200-lenovo] For historical cards with public list prices, I convert purchase price into an implied GPU-hour by amortizing the card over three years at 100% utilization. For current cloud cards, I use Lambda's listed on-demand price per GPU-hour. C870 pricing comes from HPCwire's 2007 Tesla launch coverage, P100/V100 list prices come from Microway's 2018 price analysis, and A100/H100/B200 hourly prices come from Lambda's instances page.[^tesla-c870-price][^p100-v100-price][^lambda-pricing]

| GPU | normalized compute used | price basis | USD/GPU-hour | TFLOP/s per USD/hour | PFLOP-s per USD |
| --- | ---: | --- | ---: | ---: | ---: |
| Tesla C870 | 0.518 TFLOP/s FP32 proxy | USD 1,499 list, 3-year amortized | 0.057 | 9.1 | 32.7 |
| Tesla P100 SXM2 | 21.2 TFLOP/s FP16 | USD 9,428 list, 3-year amortized | 0.359 | 59.1 | 212.7 |
| Tesla V100 SXM | 125 TFLOP/s Tensor | USD 10,664 list, 3-year amortized | 0.406 | 308.0 | 1,109.0 |
| A100 SXM 80GB | 312 TFLOP/s FP16/BF16 Tensor | Lambda 8x A100 SXM price | 2.79 | 111.8 | 402.6 |
| H100 SXM 80GB | 989 TFLOP/s dense FP16/BF16 Tensor | Lambda 8x H100 SXM price | 3.99 | 247.9 | 892.3 |
| B200 SXM6 | 2,250 TFLOP/s dense FP16/BF16 Tensor | Lambda 8x B200 SXM6 price | 6.69 | 336.3 | 1,210.8 |

This is still a rough engineering estimate, not a purchasing benchmark. The utilization assumption is optimistic for owned hardware, cloud prices include more than the GPU chip, and C870 is only a legacy proxy. The normalized shape is still useful: FP16-class compute improved enormously, but compute per dollar improved in uneven steps rather than as a smooth free lunch. Epoch AI's broader historical work reaches the same qualitative conclusion: GPU FLOP/s per dollar doubled roughly every 2.5 years across 2006-2021, and its newer AI hardware trend page estimates AI chip performance per dollar improving by about 37% per year across 2012-2025.[^gpu-price-performance][^epoch-ai-trends] Our World in Data republishes the same broad compute-per-dollar series as an interactive chart, adjusted for inflation.[^owid-gpu-price-performance]

<figure class="post-figure">
  <img src="{{ '/assets/gpu-compute-evolution.svg' | relative_url }}" alt="Two-panel log-scale chart titled GPU compute rose faster than compute per dollar, comparing FP16-normalized NVIDIA GPU compute with real estimated GPU compute-per-dollar points.">
  <figcaption>Concrete FP16-normalized price-performance estimates make the slowdown point sharper: raw math still jumps, but delivered compute per dollar depends on price, utilization, and cloud economics.</figcaption>
</figure>

#### 2.1 ALU manufacturing: narrow math buys more lanes

The ALU-level version of the story is simpler. Arithmetic got cheaper because accelerators stopped treating every operation as a wide general-purpose floating-point operation. A lower-bound manufacturing proxy is:

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

| operation at 45nm | area | units per mm2 | raw cost per 1M units | area advantage |
| --- | ---: | ---: | ---: | ---: |
| 16-bit FP add | 1,360 um2 | 735 | USD 38 | 3.1x vs FP32 add |
| 32-bit FP add | 4,184 um2 | 239 | USD 118 | baseline |
| 16-bit FP multiply | 1,640 um2 | 610 | USD 46 | 4.7x vs FP32 multiply |
| 32-bit FP multiply | 7,700 um2 | 130 | USD 218 | baseline |

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

| node | era | logic-density reference | estimated FP16 mul+add area | area shrink vs 45nm | units per mm2 | raw cost per 1M units |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 45nm | 2007 | 6.25 MTr/mm2 baseline | 3,000 um2 | 1.0x | 333 | USD 85 |
| 28nm | 2010 | 15.3 MTr/mm2 | 1,225 um2 | 2.4x | 816 | USD 52 |
| 16nm | 2015 | 28.9 MTr/mm2 | 649 um2 | 4.6x | 1,541 | USD 37 |
| 7nm | 2018 | 91.2 MTr/mm2 | 206 um2 | 14.6x | 4,864 | USD 27 |
| 5nm | 2020 | 138.2 MTr/mm2 | 136 um2 | 22.1x | 7,371 | USD 33 |
| 3nm | 2024 | 216 MTr/mm2 | 87 um2 | 34.6x | 11,520 | USD 24 |
| N2 estimate | 2026 | 313 MTr/mm2 | 60 um2 | 50.1x | 16,693 | USD 25 |
| 18A estimate | 2026 | 238 MTr/mm2 | 79 um2 | 38.1x | 12,693 | USD 33 |

The area trend is the important signal: a 16-bit floating-point multiply-plus-add datapath that is about 3,000 um2 at 45nm becomes an order of magnitude smaller by 7nm, roughly 35x smaller by 3nm, and about 50x smaller at an N2-style next node under a pure logic-density scaling model. The raw wafer-cost proxy falls less smoothly because advanced wafer prices rise sharply. N2 can fit more arithmetic, but a USD 30,000 wafer can erase much of the dollar-per-ALU gain versus 3nm. Real ALUs also need registers, operand routing, control, clocking, SRAM, verification margin, and yield. Tensor cores improve the economics further by amortizing control and data movement across matrix tiles instead of treating every multiply-add as an isolated scalar unit.

<figure class="post-figure">
  <img src="{{ '/assets/alu-dollar-cost-trend.svg' | relative_url }}" alt="Line chart showing estimated raw FP16 multiply-add ALU cost per one million units from 45nm through N2 and Intel 18A.">
  <figcaption>Logic density keeps shrinking the raw FP16 arithmetic datapath, but wafer prices flatten the dollar-cost curve at leading-edge nodes.</figcaption>
</figure>

### 3. Memory hierarchy: bytes have different economics

Memory and manufacturing show the same pattern. Compute can keep rising, but every token also needs bytes close to the math unit. The difficult part is that each level of memory optimizes a different constraint: on-chip SRAM is fast but area-expensive, HBM is bandwidth-rich but package-expensive, commodity DRAM is capacity-rich but far away, and advanced wafers are no longer getting cheap fast enough to hide the tradeoff.

#### 3.1 On-chip SRAM: fast bytes are area-limited

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

| node | SRAM bitcell | wafer price used | raw MB area | raw SRAM cost proxy |
| --- | ---: | ---: | ---: | ---: |
| 28nm | 0.127 um2 | USD 3,000 | 1.065 mm2/MB | USD 0.045/MB |
| 7nm | 0.027 um2 | USD 9,346 | 0.226 mm2/MB | USD 0.030/MB |
| 5nm | 0.021 um2 | USD 16,988 | 0.176 mm2/MB | USD 0.042/MB |
| 3nm / 2nm-class | 0.021 um2 | USD 19,500 | 0.176 mm2/MB | USD 0.049/MB |

The punchline is not that SRAM got worse in absolute density. It got much denser. The punchline is that after 7nm, bitcell shrink is small while wafer cost keeps rising. That means larger on-chip caches require more deliberate architectural justification: more L2, more shared memory, tensor memory, larger register files, and better reuse have to earn their silicon area.

NVIDIA GPU caches show the architectural response. P100 had about 4 MB of L2, V100 6 MB, A100 40 MB, H100 50 MB, and public B200 analysis reports about 126 MB of total L2.[^a100][^h100][^chips-b200-cache] More on-chip SRAM is being used because going to HBM is expensive in energy and latency, but the amount is still tiny compared with model state and KV cache.

#### 3.2 Off-chip memory: capacity and bandwidth diverge

Off-chip memory has split into two worlds. Commodity DRAM remains the capacity workhorse, but its price-per-GB improvement slowed sharply after 2010. Stanford DAM's compiled memory-price dataset shows cheapest DRAM falling from about USD 185/GB in 2005 to USD 12.2/GB in 2010, then only to USD 3.0/GB by 2020 and about USD 3.45/GB in July 2026.[^stanford-memory-prices] HBM moves in the other direction: it is not cheap capacity, it is purchased bandwidth close to the accelerator. Rambus summarizes HBM's speed evolution from 128 GB/s per HBM device to 2.048 TB/s for HBM4, while Stanford DAM's modeled HBM data puts HBM2e around USD 6/GB and HBM3e peak around USD 18/GB.[^rambus-hbm][^stanford-memory-prices]

| off-chip memory trend | representative anchors |
| --- | --- |
| GPU HBM capacity | P100: 16 GB; V100: 16 GB; A100: 40 GB; H100: 80 GB; B200: 180 GB[^p100][^v100][^a100][^h100][^b200-lenovo] |
| GPU HBM bandwidth | P100: 720 GB/s; V100: 900 GB/s; A100: 1,555 GB/s; H100: over 3 TB/s; B200: 7.7 TB/s[^p100][^v100][^a100][^h100][^b200-lenovo] |
| commodity DRAM price/capacity | about USD 185/GB in 2005, USD 12.2/GB in 2010, USD 3.0/GB in 2020, and USD 3.45/GB in July 2026[^stanford-memory-prices] |
| HBM price/capacity | HBM2e around USD 6/GB, HBM3 around USD 9/GB, HBM3e peak around USD 18/GB, HBM4 projected around USD 16.5/GB[^stanford-memory-prices] |
| HBM price/bandwidth | HBM2e around USD 209 per TB/s, HBM3 around USD 264 per TB/s, HBM3e peak around USD 352 per TB/s, HBM4 projected around USD 297 per TB/s[^stanford-memory-prices] |

#### 3.3 Manufacturing: wafer cost pushes back

The manufacturing layer is the shared denominator under both compute and SRAM. If each wafer gets more expensive, every large die, cache expansion, interposer choice, and yield loss has a higher dollar impact.

| process node | approximate era | wafer price anchor | why it matters |
| --- | ---: | ---: | --- |
| 28nm | 2010 | USD 3,000 | cheap enough that SRAM scaling still translated into lower raw MB cost |
| 7nm | 2018 | USD 9,346 | density improved, but wafer price more than tripled |
| 5nm | 2020 | USD 16,988 | raw SRAM cost proxy rose again despite smaller cells |
| 3nm | 2024 | USD 19,500 | wafer price keeps rising while SRAM bitcell shrink slows |

<figure class="post-figure">
  <img src="{{ '/assets/memory-manufacturing-trends.svg' | relative_url }}" alt="Four-panel chart showing SRAM cost proxy, commodity DRAM price per GB, HBM price per bandwidth, and wafer price by process node.">
  <figcaption>Memory economics explain why AI infra is increasingly about locality: on-chip SRAM density is harder to buy with node shrinks, HBM bandwidth is expensive capacity, commodity DRAM is cheap but far away, and advanced wafer prices keep rising.</figcaption>
</figure>

### 4. Interconnection and communication: scale-out has a tax

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

| parallelism pattern | communication pressure |
| --- | --- |
| data parallel | gradient all-reduce or reduce-scatter / all-gather |
| tensor parallel | activation all-reduce and all-gather inside layers |
| pipeline parallel | boundary activations and pipeline bubbles |
| expert parallel / MoE | token dispatch and all-to-all routing |
| disaggregated serving | KV cache movement, prefill/decode handoff, remote memory access |

Interconnect bandwidth is improving aggressively because this tax is now first-order. NVIDIA lists NVLink bandwidth per GPU rising from 900 GB/s on Hopper to 1.8 TB/s on Blackwell and 3.6 TB/s on Rubin; its NVLink Switch table lists NVL72 aggregate bandwidth rising from 130 TB/s on Blackwell to 260 TB/s on Rubin.[^nvlink] NVIDIA's HGX Rubin page makes the same point at the system level: higher token throughput is tied not only to more NVFP4 compute, but also to more HBM bandwidth and more NVLink Switch bandwidth.[^hgx-rubin]

| system generation | interconnect anchor | why it matters |
| --- | ---: | --- |
| Hopper | 900 GB/s NVLink per GPU | scale-up communication becomes part of model throughput |
| Blackwell | 1.8 TB/s NVLink per GPU; 130 TB/s NVL72 aggregate | larger rack-scale GPU domains for model parallelism |
| Rubin | 3.6 TB/s NVLink per GPU; 260 TB/s NVL72 aggregate | communication bandwidth has to scale with MoE, long context, and agentic inference |

There are two different network curves hiding behind the same word "interconnect":

- **Scale-up** is the tightly coupled GPU domain inside a box or rack. NVLink / NVSwitch bandwidth is not sold like a generic switch port; it is bundled into GPU systems, board design, power delivery, and thermal design. The useful public metric is bandwidth per GPU or per rack-scale domain.
- **Scale-out** is the cluster fabric across nodes and racks. Ethernet and InfiniBand have visible port speeds, switch radix, optics, cables, NICs, and sometimes observable street prices. This is where a rough dollar-per-Gb/s proxy is possible.

The speed curve is steep. InfiniBand moved from 4x QDR at 32 Gb/s in the late 2000s, to EDR 100 Gb/s, HDR 200 Gb/s, NDR 400 Gb/s, and XDR 800 Gb/s. Ethernet followed the same broad shape: 40/100GbE was standardized in 2010, 200/400GbE in 2017, and 800GbE in 2024.[^infiniband-rates][^ethernet-100g][^ethernet-400g][^ethernet-800g] NVIDIA's current Quantum-X800 documentation lists 72-port and 144-port XDR systems at 800 Gb/s per port, up to 115.2 Tb/s of maximum throughput for the 4U system.[^quantum-x800]

For a dollar-per-speed proxy, use switch chassis price divided by front-panel bandwidth. This is not total cluster networking cost. It excludes optics, cables, NICs, support contracts, power, rack layout, and topology oversubscription. It is still useful because it shows the direction of the switching layer itself. The anchors below combine a legacy Cisco 10GbE price-list snapshot, public Mellanox / NVIDIA InfiniBand switch listings, and current SN5610 800GbE listing/spec data.[^nexus-price][^mellanox-switch-prices][^sn5610-price][^sn5610-specs]

| scale-out switch proxy | approximate era | ports x port speed | public price anchor | switch dollars per Gb/s |
| --- | ---: | ---: | ---: | ---: |
| Cisco Nexus 5020 10GbE | 2008 | 40 x 10 Gb/s | USD 28,770 list-price snapshot | USD 72/Gb/s |
| Mellanox SB7800 EDR | 2015 | 36 x 100 Gb/s | USD 10,259 channel listing | USD 2.85/Gb/s |
| Mellanox QM8700 HDR | 2018 | 40 x 200 Gb/s | USD 18,740 channel listing | USD 2.34/Gb/s |
| NVIDIA QM9700 NDR | 2022 | 64 x 400 Gb/s | USD 32,870 channel listing | USD 1.28/Gb/s |
| NVIDIA SN5610 800GbE | 2026 | 64 x 800 Gb/s | USD 51,999 channel listing | USD 1.02/Gb/s |

<figure class="post-figure">
  <img src="{{ '/assets/interconnect-speed-cost-trend.svg' | relative_url }}" alt="Dual-axis chart showing scale-out switch port speed rising from 10G to 800G while switch dollar per gigabit falls from about 72 dollars per gigabit to about one dollar per gigabit.">
  <figcaption>Scale-out switch bandwidth improved dramatically over the last two decades, but the system-level network bill does not fall as fast as raw switch dollars per Gb/s because optics, NICs, cables, power, and topology complexity become first-order costs.</figcaption>
</figure>

The punchline is subtle: network silicon has delivered a large cost-per-bit improvement, but AI clusters keep spending the savings. Higher port speed enables larger all-reduce domains, more tensor-parallel shards, more MoE all-to-all traffic, and more disaggregated serving. The value of better AI infrastructure is therefore not just "buy faster switches." It is reducing bytes moved, placing bytes closer to compute, overlapping collectives with kernels, and choosing parallelism plans that turn expensive network bandwidth into useful tokens.

This is why "chips are slowing down" is not only a FLOP story. It is a locality and communication story. When model weights, activations, KV cache, and tool-use context grow, the system pays for bytes in several currencies: SRAM area, HBM dollars, HBM bandwidth, interconnect bandwidth, synchronization time, package complexity, wafer cost, and energy. Good AI infrastructure wins by spending fewer bytes, reusing them closer to compute, and making expensive memory and network bandwidth do useful work more often.

The token is the economic unit
---

The token is a useful accounting unit because it connects the full stack:

```text
token cost ~= model math + memory movement + cache residency + communication + scheduling overhead + verification/retry overhead
```

That is why low-level details matter:

- Faster attention changes long-context serving cost.
- Better quantization changes memory pressure and batch size.
- Better batching changes utilization and latency tails.
- Better routing changes expert-model cost.
- Better cache management changes context feasibility.
- Better interconnect and collective scheduling changes scale-out efficiency.
- Better compiler lowering changes which model shapes are practical.
- Better parallelism changes whether training runs are stable and affordable.

The next AI infra frontier is not simply larger clusters. It is a tighter control loop between algorithms, model architecture, serving systems, kernels, compilers, and chips. The teams that win will not only have better models or better hardware. They will have better token economics.

References
---

[^openai-compute]: OpenAI, [AI and Compute](https://openai.com/index/ai-and-compute/), 2018.
[^epoch-compute]: Jaime Sevilla et al., [Compute Trends Across Three Eras of Machine Learning](https://arxiv.org/abs/2202.05924), 2022.
[^kaplan]: Jared Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), 2020.
[^chinchilla]: Jordan Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), 2022.
[^large-language-monkeys]: Bradley Brown et al., [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787), 2024.
[^sutter]: Herb Sutter, [The Free Lunch Is Over](https://www.cs.utexas.edu/~lin/cs380p/Free_Lunch.pdf), 2005.
[^dark-silicon]: Hadi Esmaeilzadeh et al., [Dark Silicon and the End of Multicore Scaling](https://research.cs.wisc.edu/vertical/papers/2011/isca11-darksilicon.pdf), 2011.
[^horowitz]: Mark Horowitz, [Computing's Energy Problem](https://gwern.net/doc/cs/hardware/2014-horowitz-2.pdf), ISSCC 2014.
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
[^rubin]: NVIDIA Newsroom, [NVIDIA Kicks Off the Next Generation of AI With Rubin](https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer), 2026.
[^rubin-cpx]: NVIDIA Newsroom, [NVIDIA Unveils Rubin CPX](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference), 2025.
[^b200-lenovo]: Lenovo Press, [ThinkSystem NVIDIA HGX B200 180GB 1000W GPU](https://lenovopress.lenovo.com/lp2226-thinksystem-nvidia-b200-180gb-1000w-gpu), accessed 2026-07-04.
[^tesla-c870-price]: Michael Feldman, [NVIDIA Takes Direct Aim at High Performance Computing](https://www.hpcwire.com/2007/06/22/nvidia_takes_direct_aim_at_high_performance_computing-1/), HPCwire, 2007.
[^p100-v100-price]: Brett Newman, [NVIDIA Tesla V100 Price Analysis](https://www.microway.com/hpc-tech-tips/nvidia-tesla-v100-price-analysis/), Microway, 2018.
[^lambda-pricing]: Lambda, [Instances](https://lambda.ai/instances), accessed 2026-07-04.
[^gpu-price-performance]: Jaime Sevilla and Pablo Villalobos, [Trends in GPU Price-Performance](https://epoch.ai/publications/trends-in-gpu-price-performance), Epoch AI, 2022.
[^epoch-ai-trends]: Epoch AI, [Trends in Artificial Intelligence: AI Hardware](https://epoch.ai/trends), accessed 2026-07-02.
[^owid-gpu-price-performance]: Our World in Data, [GPU computational performance per dollar](https://ourworldindata.org/grapher/gpu-price-performance), accessed 2026-07-02.
[^tsmc-28nm-sram]: Mark LaPedus, [TSMC devises SRAM cell at 28-nm](https://www.eetimes.com/tsmc-devises-sram-cell-at-28-nm/), EE Times, 2009.
[^tsmc-5nm-sram]: SemiWiki, [TSMC's 5nm 0.021um2 SRAM Cell Using EUV and High Mobility Channel with Write Assist at ISSCC2020](https://semiwiki.com/semiconductor-manufacturers/tsmc/283487-tsmcs-5nm-0-021um2-sram-cell-using-euv-and-high-mobility-channel-with-write-assist-at-isscc2020/), 2020.
[^tsmc-2nm-sram]: TSMC Research, [Memory publications](https://research.tsmc.com/english/research/memory/publish-time-1.html), accessed 2026-07-04.
[^cset-wafer-cost]: Center for Security and Emerging Technology, [Analysts believe that a single TSMC 5nm wafer costs USD 17,000](https://cset.georgetown.edu/article/analysts-believe-that-a-single-tsmc-5nm-wafer-costs-17000/), 2020.
[^wafer-pricing]: Silicon Analysts, [Semiconductor Wafer Pricing by Process Node](https://siliconanalysts.com/data/wafer-pricing), accessed 2026-07-04.
[^chips-b200-cache]: Chips and Cheese, [Nvidia's B200: Keeping the CUDA Juggernaut Rolling](https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut), 2025.
[^stanford-memory-prices]: David Shim, Stanford DAM, [Memory Prices](https://dam.stanford.edu/memory-prices.html), accessed 2026-07-04.
[^rambus-hbm]: Rambus, [High Bandwidth Memory: Everything You Need to Know](https://www.rambus.com/blogs/hbm3-everything-you-need-to-know/), updated 2026.
[^alu-area-cost]: Ting-Yu Yeh, [Accelerator Architectures for Machine Learning](https://people.cs.nycu.edu.tw/~ttyeh/course/2023_Fall/IOC5009/slide/lecture-3.pdf), lecture slides citing Horowitz ISSCC 2014 operation energy and area data, accessed 2026-07-04.
[^cmos-cost]: Tim Johnson, [CMOS Cost](https://faculty-web.msoe.edu/johnsontimoj/EE4980/files4980/cmos_cost.pdf), MSOE EE 4980 notes, accessed 2026-07-04.
[^logic-density-28-7]: Team VLSI, [TSMC 7nm, 16nm and 28nm Technology node comparisons](https://teamvlsi.com/2021/09/tsmc-7nm-16nm-and-28nm-technology-node-comparisons.html), 2021.
[^process-density-5nm]: Wikipedia, [5 nm process](https://en.wikipedia.org/wiki/5_nm_process), accessed 2026-07-04.
[^process-density-3nm]: Wikipedia, [3 nm process](https://en.wikipedia.org/wiki/3_nm_process), accessed 2026-07-04.
[^next-node-density]: Anton Shilov, [Intel's 18A and TSMC's N2 process nodes compared](https://www.tomshardware.com/tech-industry/intels-18a-and-tsmcs-n2-process-nodes-compared-intel-is-faster-but-tsmc-is-denser), Tom's Hardware, 2025.
[^n2-wafer-price]: Astute Group, [TSMC's 2nm Wafer Price Hits USD 30,000 Amid Monopoly Concerns](https://www.astutegroup.com/news/industrial/tsmcs-2nm-wafer-price-hits-30000-amid-monopoly-concerns/), 2025.
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
