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

In strong scaling, the workload is fixed and more hardware is added to finish sooner. The limiting terms are communication, synchronization, stragglers, and idle time. Adding more devices eventually exposes imperfect overlap and coordination overhead.

Pre-LLM consumer infrastructure mostly fits this pattern. A video stream, social-network feed, or game session can be expensive at global scale, but the compute per user is usually bounded by the product workload:

```text
pre-LLM app:
  per_user_compute ~= O(1)
  total_compute(N users) ~= N * O(1)

strong scaling goal:
  fixed workload W
  add hardware P
  reduce latency: time ~= W / P + communication_overhead(P)
```

In other words, once the product behavior is fixed, infra optimization mostly reduces the constant factor for serving the same request, stream, feed, or frame. The user count grows, but the compute size of each user's unit of work does not keep expanding because the model got larger or the context got longer.

LLM infrastructure is different. The unit workload itself scales with model size and token length:

```text
LLM request, rough dense-transformer intuition:
  P = model parameters
  T = input_tokens + output_tokens

  per_request_compute ~= O(P * T) + attention/cache terms

post-LLM trend:
  P increases
  T increases
  inference-time steps increase
  therefore per_request_compute is not O(1)
```

This is why LLMs fit weak scaling better. In weak scaling, the hardware grows and the workload grows with it. More compute is used to train larger models, consume more data, extend context length, add modalities, and spend more compute at inference time. The bottleneck moves instead of disappearing:

- More GPUs make interconnect topology and collective overlap important.
- Longer context turns KV cache capacity and bandwidth into serving constraints.
- Larger batches improve throughput but change latency and memory pressure.
- MoE and routing improve parameter efficiency but add load-balancing problems.
- Test-time compute improves quality but increases token budget variance.

The financial implication is the important part. If an infra optimization reduces unit cost by a factor `r`, then:

```text
baseline_cost = requests * workload_per_request * cost_per_compute
optimized_cost = requests * workload_per_request * cost_per_compute * r
savings = requests * workload_per_request * cost_per_compute * (1 - r)
```

Under strong scaling, `workload_per_request ~= O(1)`, so savings mostly track traffic volume. Under weak scaling, `workload_per_request` grows with model size, context length, generated tokens, and agent steps. The same 20% kernel, compiler, cache, or batching win is therefore magnified by the growing workload. This is the "the more you buy, the more you save" property of AI infra: as token demand scales up, every percentage point of efficiency converts into a larger absolute dollar saving.

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

Compute per dollar improved too, but at a steadier rate. Epoch AI estimated that GPU FLOP/s per dollar doubled roughly every 2.5 years across 2006-2021, and roughly every 2.07 years for GPUs commonly used in ML research.[^gpu-price-performance] Their newer AI hardware trend page estimates AI chip performance per dollar improving by about 37% per year across 2012-2025.[^epoch-ai-trends] Our World in Data publishes the same broad compute-per-dollar series as an interactive chart, adjusted for inflation.[^owid-gpu-price-performance]

<figure class="post-figure">
  <img src="{{ '/assets/gpu-compute-evolution.svg' | relative_url }}" alt="Log-scale chart comparing representative NVIDIA GPU peak compute with an Epoch AI GPU compute per dollar trend, normalized to 2007.">
  <figcaption>Peak AI math rates rose dramatically as GPUs adopted lower-precision tensor paths; compute per dollar improved too, but at a steadier hardware-economics trend.</figcaption>
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
[^gpu-price-performance]: Jaime Sevilla and Pablo Villalobos, [Trends in GPU Price-Performance](https://epoch.ai/publications/trends-in-gpu-price-performance), Epoch AI, 2022.
[^epoch-ai-trends]: Epoch AI, [Trends in Artificial Intelligence: AI Hardware](https://epoch.ai/trends), accessed 2026-07-02.
[^owid-gpu-price-performance]: Our World in Data, [GPU computational performance per dollar](https://ourworldindata.org/grapher/gpu-price-performance), accessed 2026-07-02.
