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

The H100 is a useful example. NVIDIA's Hopper material highlights HBM3 bandwidth around 3 TB/s, a 50 MB L2 cache, Transformer Engine support, NVLink/NVSwitch scale-out, and low-precision tensor paths.[^h100] These are not just "more FLOPs." They are area, power, and system-design choices that help specific workload shapes.

The GPU timeline shows the same pattern:

| generation | headline compute direction |
| --- | --- |
| Tesla C870 / C1060 | CUDA-era single-precision throughput |
| K20X | higher FP32/FP64 HPC throughput |
| P100 / V100 | FP16 and Tensor Core acceleration |
| A100 / H100 | BF16, TF32, FP8, sparsity, larger memory systems |
| Blackwell / Rubin | FP4/NVFP4 and rack-scale AI systems |

Representative figures make the jump visible. Tesla C870 was advertised at 518 GFLOP/s peak single precision in 2007.[^tesla-c870] C1060 reached 933 GFLOP/s in 2008.[^tesla-c1060] K20X reached 3.95 TFLOP/s single precision and 1.31 TFLOP/s double precision in 2012.[^tesla-k20x] P100 delivered 21.2 TFLOP/s FP16 in 2016, V100 delivered about 125-130 Tensor TFLOP/s, A100 reached 312 TFLOP/s dense FP16/BF16 Tensor Core performance, and H100 lists 1,979 TFLOP/s FP16/BF16 Tensor Core and 3,958 TFLOP/s FP8 Tensor Core performance.[^p100][^v100][^a100][^h100-spec]

Blackwell continues the shift. NVIDIA's DGX B200 system lists 144 PFLOP/s FP4 Tensor Core performance across eight Blackwell GPUs, roughly 18 PFLOP/s per GPU at the published system level, while Blackwell Ultra emphasizes 15 PFLOP/s dense NVFP4 per GPU.[^blackwell-b200][^blackwell-ultra] NVIDIA's Rubin announcement lists 50 PFLOP/s NVFP4 inference compute per Rubin GPU, and Vera Rubin NVL144 CPX is framed around 8 exaFLOP/s of rack-scale AI performance for massive-context inference.[^rubin][^rubin-cpx]

These figures are not an apples-to-apples speedup curve. The datatype, sparsity mode, memory system, and programming model all changed. That is the important part. GPU progress came from changing the numerical contract: CUDA, SIMT execution, HBM, NVLink, tensor cores, TF32, BF16, FP8, FP4, sparsity, and compiler/runtime support made model structure visible to hardware.

To make the price-performance curve concrete, use a simple rental-equivalent metric:

<div class="math-block">
$$
\Pi = \frac{\text{peak TFLOP/s}}{\text{USD per GPU-hour}},
\qquad
\text{PFLOP-s per USD} = 3.6\Pi
$$
</div>

One TFLOP/s sustained for one hour produces 3.6 PFLOP-s. For historical cards with public list prices, I convert purchase price into an implied GPU-hour by amortizing the card over three years at 100% utilization. For current cloud cards, I use Lambda's listed on-demand price per GPU-hour. C870 pricing comes from HPCwire's 2007 Tesla launch coverage, P100/V100 list prices come from Microway's 2018 price analysis, and A100/H100/B200 hourly prices come from Lambda's instances page.[^tesla-c870-price][^p100-v100-price][^lambda-pricing]

| GPU | peak compute used | price basis | USD/GPU-hour | TFLOP/s per USD/hour | PFLOP-s per USD |
| --- | ---: | --- | ---: | ---: | ---: |
| Tesla C870 | 0.518 TFLOP/s FP32 | USD 1,499 list, 3-year amortized | 0.057 | 9.1 | 32.7 |
| Tesla P100 SXM2 | 21.2 TFLOP/s FP16 | USD 9,428 list, 3-year amortized | 0.359 | 59.1 | 212.7 |
| Tesla V100 SXM | 125 TFLOP/s Tensor | USD 10,664 list, 3-year amortized | 0.406 | 308.0 | 1,109.0 |
| A100 SXM 80GB | 312 TFLOP/s FP16/BF16 Tensor | Lambda 8x A100 SXM price | 2.79 | 111.8 | 402.6 |
| H100 SXM 80GB | 3,958 TFLOP/s FP8 Tensor | Lambda 8x H100 SXM price | 3.99 | 992.0 | 3,571.1 |
| B200 SXM6 | 18,000 TFLOP/s FP4 Tensor | Lambda 8x B200 SXM6 price | 6.69 | 2,690.6 | 9,686.1 |

This is still a rough engineering estimate, not a purchasing benchmark. The datatype changes across generations, the utilization assumption is optimistic for owned hardware, and cloud prices include more than the GPU chip. The shape is still useful: peak AI math has moved from sub-TFLOP FP32 to multi-PFLOP low-precision tensor paths, while compute per dollar improved in uneven steps rather than as a smooth free lunch. Epoch AI's broader historical work reaches the same qualitative conclusion: GPU FLOP/s per dollar doubled roughly every 2.5 years across 2006-2021, and its newer AI hardware trend page estimates AI chip performance per dollar improving by about 37% per year across 2012-2025.[^gpu-price-performance][^epoch-ai-trends] Our World in Data republishes the same broad compute-per-dollar series as an interactive chart, adjusted for inflation.[^owid-gpu-price-performance]

<figure class="post-figure">
  <img src="{{ '/assets/gpu-compute-evolution.svg' | relative_url }}" alt="Two-panel log-scale chart comparing representative NVIDIA GPU peak compute and estimated compute per dollar.">
  <figcaption>Concrete price-performance estimates make the slowdown point sharper: raw peak math still jumps, but delivered compute per dollar depends on price, utilization, datatype, and cloud economics.</figcaption>
</figure>

The token is the economic unit
---

The token is a useful accounting unit because it connects the full stack:

```text
token cost ~= model math + memory movement + cache residency + scheduling overhead + verification/retry overhead
```

That is why low-level details matter:

- Faster attention changes long-context serving cost.
- Better quantization changes memory pressure and batch size.
- Better batching changes utilization and latency tails.
- Better routing changes expert-model cost.
- Better cache management changes context feasibility.
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
[^tesla-c870-price]: Michael Feldman, [NVIDIA Takes Direct Aim at High Performance Computing](https://www.hpcwire.com/2007/06/22/nvidia_takes_direct_aim_at_high_performance_computing-1/), HPCwire, 2007.
[^p100-v100-price]: Brett Newman, [NVIDIA Tesla V100 Price Analysis](https://www.microway.com/hpc-tech-tips/nvidia-tesla-v100-price-analysis/), Microway, 2018.
[^lambda-pricing]: Lambda, [Instances](https://lambda.ai/instances), accessed 2026-07-04.
[^gpu-price-performance]: Jaime Sevilla and Pablo Villalobos, [Trends in GPU Price-Performance](https://epoch.ai/publications/trends-in-gpu-price-performance), Epoch AI, 2022.
[^epoch-ai-trends]: Epoch AI, [Trends in Artificial Intelligence: AI Hardware](https://epoch.ai/trends), accessed 2026-07-02.
[^owid-gpu-price-performance]: Our World in Data, [GPU computational performance per dollar](https://ourworldindata.org/grapher/gpu-price-performance), accessed 2026-07-02.
