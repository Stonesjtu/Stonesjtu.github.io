---
layout: post
title: "[WIP] AI infra and tokenomics"
topic: "AI infrastructure"
sequence: 8
source_url: https://app.notion.com/p/38d2ec4bb1f0808ea061d11de43d93a6
source_label: "Original outline on Notion"
excerpt: "Why AI infrastructure has become the scarce layer of the modern stack, and why token economics increasingly depends on model, systems, and chip co-design."
---

Every generation of computing has an infrastructure layer that quietly decides what is possible. In web infrastructure, the scarce resource was reliable request handling at scale. In mobile, it was distribution, latency, battery, and cloud synchronization. In machine learning, the scarce resource became data movement and accelerator utilization. In large language models, the scarce resource is now tokens: how many can be trained, served, routed, cached, and generated per unit of capital, power, and engineering time.

That is why AI infrastructure feels different from conventional infrastructure. It is not just a support function around the product. It is part of the product's economics. The cost of a token is determined by the whole stack: model architecture, training recipe, serving system, framework, compiler, kernel, accelerator backend, memory hierarchy, networking, chip design, packaging, and manufacturing. A small inefficiency at any layer can multiply across trillions of tokens.

There is a simple reason this layer has become so valuable: demand is compounding on the model side while supply is getting harder on the hardware side. OpenAI's 2018 analysis found that compute used in the largest AI training runs had been doubling every 3.4 months from 2012 onward, far faster than the usual Moore's law comparison point.[^openai-compute] Epoch AI's later study gives a more conservative but still aggressive picture: since the early deep learning era, training compute for milestone ML systems has doubled roughly every 5 to 6 months.[^epoch-compute]

<figure class="post-figure">
  <img src="{{ '/assets/ai-infra-demand-supply.svg' | relative_url }}" alt="Conceptual chart showing AI model and token demand rising faster than accelerator compute, memory bandwidth, and power budgets.">
  <figcaption>AI infrastructure matters because the curve to serve is steeper than the curve hardware gives us for free.</figcaption>
</figure>

From conventional infra to agentic infra
---

Conventional infrastructure was built around services. The central problems were availability, isolation, deployment, storage, observability, and request throughput. The unit of work was usually a request, a job, or a transaction. Good infrastructure made software easier to operate.

Machine learning infrastructure changed the unit of work. Instead of serving deterministic business logic, the system had to move tensors through training and inference pipelines. The scarce resources became GPU hours, memory bandwidth, distributed data loading, experiment reproducibility, and model artifact management. Good infrastructure made model iteration possible.

Deep learning infrastructure made the stack more specialized. GPUs were no longer optional accelerators; they became the main computer. Frameworks had to express large computation graphs, schedule kernels, manage autograd, and coordinate distributed training. Performance no longer came from simply adding machines. It came from keeping accelerators fed.

LLM infrastructure tightened the loop even further. Pretraining pushed cluster scale, interconnect bandwidth, checkpointing, parallelism strategies, and failure recovery to the center. Inference then became its own systems problem: KV cache management, batching, prefill/decode separation, speculative decoding, quantization, routing, memory pressure, and latency SLOs. The model is no longer a file that sits behind a service. It is a dynamic workload that reshapes the service around itself.

Agentic infrastructure adds another layer of pressure. The system is not only generating one response. It may plan, call tools, read documents, write code, run tests, search memory, retry failed steps, and coordinate across many model calls. This increases the number of tokens consumed per task, but it also changes the workload shape. The value is no longer just in cheaper single-turn inference. It is in reducing the total tokens and wall-clock time needed to complete useful work.

Why infra became so resource-demanding
---

The easiest way to see the pressure is through scaling. In strong scaling, the workload is fixed and we add more hardware to finish it faster. This is hard because communication, synchronization, and idle time eventually dominate. The larger the distributed system, the more expensive every imperfect overlap becomes.

In weak scaling, the hardware grows and the workload grows with it. This has been the dominant story of modern AI. We do not merely train the same model faster. We train larger models, on more data, with longer context, more modalities, more inference-time compute, and more complicated serving behavior. The appetite expands to consume the available machine.

That makes AI infrastructure unusually resource-demanding. More compute creates the opportunity for better models, but also creates new bottlenecks. More GPUs require better networking. Better networking exposes memory pressure. More memory makes larger batches possible, which changes kernel efficiency. Larger models need new parallelism strategies. Longer context turns KV cache into a first-class systems problem. More inference traffic forces batching and scheduling to become economic decisions.

The scaling-law literature explains why this appetite is rational rather than just fashionable. Kaplan et al. found language-model loss following power laws with model size, dataset size, and training compute across many orders of magnitude.[^kaplan] Hoffmann et al. then showed that compute-optimal training needs model size and training tokens to scale together; their Chinchilla result used the same compute budget as Gopher but trained a smaller model on more tokens, improving quality while reducing downstream inference cost.[^chinchilla] The practical lesson is not "make everything bigger blindly." It is that parameters, data, training compute, and inference cost are coupled.

In this regime, infrastructure is not a static foundation. It is a moving frontier.

The AI infra ecosystem
---

The AI infrastructure ecosystem is broad because the workload cuts across so many boundaries. Algorithm designers decide which learning signals, objectives, and inference strategies are worth pursuing. Model designers choose architectures that trade off quality, compute, memory, and serving complexity. Infrastructure engineers turn those choices into reliable systems. Serving engineers make latency, throughput, batching, and cache behavior work under real traffic.

Framework engineers provide the programming model. Kernel engineers make the hot paths fast. Compiler engineers lower high-level tensor programs into hardware-efficient execution. Accelerator backend engineers expose new hardware capabilities to the software stack. Chip designers decide what compute, memory, interconnect, and data types the next generation can afford. Chip manufacturers determine which of those designs can be produced at scale.

No single layer owns the final cost. A model architecture that looks elegant on paper can be expensive to serve. A chip feature that looks powerful can be hard to reach from frameworks. A compiler optimization can be blocked by dynamic shapes. A serving strategy can improve throughput but hurt latency. A kernel can be locally optimal and still lose if it increases memory pressure elsewhere.

This is why the most valuable work often happens between layers.

<figure class="post-figure">
  <img src="{{ '/assets/ai-infra-token-stack.svg' | relative_url }}" alt="Stack diagram showing algorithm, model, serving, framework, compiler, kernel, accelerator, memory, network, chip design, and manufacturing all contributing to token economics.">
  <figcaption>The cost and latency of one useful token is the sum of many cross-layer decisions.</figcaption>
</figure>

Infra engineering is the exciting layer
---

AI infrastructure engineering is exciting because it sits where the constraints meet. The model wants to scale. The chip no longer scales for free. Memory technology has been slowing for decades. Power and thermal limits are real. Manufacturing shrinkage is no longer the automatic economic engine it once was. Cost per FLOP does not collapse fast enough to hide inefficient software.

For a long time, the industry could rely on a convenient rhythm: chips got denser, faster, and cheaper, and software could spend some of that improvement on abstraction. AI has broken that comfort. The models are scaling faster than the hardware economics can absorb. Parameters still matter. Training FLOPs still matter. Inference FLOPs increasingly matter. Longer reasoning, tool use, search, and test-time compute all push more demand into the serving stack.

If models keep scaling while chips and memory slow down, the only practical solution is co-optimization. Model, infrastructure, and chip need to be designed together.

This is not just a story about peak FLOP/s. Sutter's "free lunch" argument captured the end of effortless single-thread speedups more than twenty years ago.[^sutter] Architecture work on dark silicon then made the power limit more explicit: even if transistors are available, power and thermal constraints prevent all of them from being active at once.[^dark-silicon] For AI systems, memory movement is often just as important. Horowitz's energy numbers are a useful reminder that moving data from memory can cost orders of magnitude more energy than simple arithmetic.[^horowitz] Modern accelerators respond with HBM, large caches, tensor cores, low-precision formats, and high-bandwidth links, but those features only help when the software stack can actually expose locality and parallelism.

This does not mean every model designer must become a chip designer, or every systems engineer must invent new architectures. It means the boundaries need to be porous. Model choices should be evaluated with serving cost in mind. Kernel and compiler work should influence which architectures are attractive. Chip features should be exposed through software early enough that model teams can use them. Serving systems should feed real workload data back into model and hardware design.

Tokenomics is the visible surface of this co-design. The question is not only how much intelligence a model has. It is how much useful intelligence can be delivered per dollar, per watt, per second, and per engineering cycle.

Models are still scaling
---

It is tempting to say that scaling is over whenever a particular curve bends or a particular benchmark saturates. But the broader trend is still that more parameters, more training compute, and more inference compute can produce more capability when used well.

More parameters can store richer representations and support broader behavior. More training FLOPs can improve the model's internal structure and generalization. More inference FLOPs can improve results through longer reasoning, search, verification, tool use, and self-correction. Agentic systems make this especially clear: the model's answer is not only the output of one forward pass, but the result of a process.

This shifts the economic question. The bottleneck is not simply whether we can train a larger model once. It is whether we can afford to deploy intelligence that spends compute at inference time. If a system can solve harder tasks by using more tokens, then token efficiency becomes a product constraint, not just an infra metric.

Inference-time scaling makes this especially concrete. Brown et al. studied repeated sampling and found that solution coverage can continue improving as sample count grows over several orders of magnitude, with automatic verification turning more samples into better measured performance in domains such as code and formal proofs.[^large-language-monkeys] That does not mean every product should sample hundreds of answers. It means serving infrastructure now participates directly in the quality curve: scheduling, verification, caching, and routing can change the capability delivered per dollar.

The best infrastructure therefore does two things at once. It lowers the cost of each token, and it reduces the number of tokens needed to achieve an outcome.

Chips are slowing down
---

The hardware side is less forgiving. Silicon is running into physical limits. Manufacturing shrinkage is harder, slower, and more expensive. Power delivery and thermal constraints make it difficult to simply add more active compute. Memory bandwidth and capacity do not improve at the same rate as compute demand. Packaging and interconnect help, but they also increase system complexity and cost.

This does not mean hardware innovation is over. It means the easy version is over. Future gains will come from specialization, packaging, memory hierarchy design, interconnect, lower precision, sparsity, compiler-visible hardware features, and better system-level balance. These gains are real, but they require software to meet the hardware halfway.

That is the central tension in AI infra. The model side wants elasticity: bigger models, longer contexts, more modalities, more reasoning, more tool calls. The chip side imposes scarcity: fixed power, finite memory, expensive bandwidth, imperfect utilization, and long hardware cycles. Infrastructure is the discipline that translates between those two realities.

The H100 is a good example of both progress and constraint. NVIDIA's Hopper material highlights HBM3 bandwidth around 3 TB/s, a 50 MB L2 cache, Transformer Engine support, NVLink/NVSwitch scale-out, and low-precision tensor paths.[^h100] Those are enormous engineering gains, but they also show the shape of the problem: the accelerator has to spend more area and system design on memory hierarchy, interconnect, and datatype specialization because raw compute alone is not enough.

The last twenty years of GPU evolution make the same point from another angle. The first CUDA-era Tesla C870 was advertised at 518 GFLOP/s peak single precision in 2007.[^tesla-c870] The C1060 reached 933 GFLOP/s in 2008.[^tesla-c1060] Kepler's K20X reached 3.95 TFLOP/s single precision and 1.31 TFLOP/s double precision in 2012.[^tesla-k20x] Then the center of gravity shifted from general FP32 throughput toward lower-precision tensor math for AI: P100 delivered 21.2 TFLOP/s FP16 in 2016, V100 delivered about 125-130 Tensor TFLOP/s, A100 reached 312 TFLOP/s dense FP16/BF16 Tensor Core performance, and H100 lists 1,979 TFLOP/s FP16/BF16 Tensor Core and 3,958 TFLOP/s FP8 Tensor Core performance.[^p100][^v100][^a100][^h100-spec]

Those numbers should not be read as one clean apples-to-apples curve, because the datatype, sparsity mode, memory system, and programming model changed. But that is exactly the infrastructure lesson. GPU progress has come not only from more scalar FLOP/s, but from changing the numerical contract: CUDA, SIMT execution, HBM, NVLink, tensor cores, TF32, BF16, FP8, sparsity, and compiler/runtime support all turned model structure into hardware efficiency. The resource trend is still impressive: Epoch AI estimated that GPU FLOP/s per dollar doubled roughly every 2.5 years across 2006-2021, and roughly every 2.07 years for GPUs commonly used in ML research.[^gpu-price-performance] Their newer AI hardware trend page estimates AI chip performance per dollar improving by about 37% per year across 2012-2025.[^epoch-ai-trends] Our World in Data publishes the same broad compute-per-dollar series as an interactive chart, adjusted for inflation.[^owid-gpu-price-performance]

The token is the economic unit
---

For LLM systems, the token is the unit where the whole stack becomes measurable. Training consumes tokens to create capability. Inference consumes tokens to deliver capability. Agentic workflows consume tokens to turn capability into completed tasks. The price of intelligence is therefore strongly tied to token throughput and token efficiency.

This is why kernel-level details matter. A faster attention kernel changes serving economics. Better quantization changes memory pressure. Better batching changes accelerator utilization. Better routing changes the cost of expert models. Better cache management changes long-context feasibility. Better compiler lowering changes whether a model design is practical. Better parallelism changes whether a training run is stable and affordable.

None of these optimizations are merely technical trivia. They decide how much intelligence can be produced from a given amount of hardware.

The next AI infra frontier is not just bigger clusters. It is a tighter feedback loop between algorithms, models, systems, and chips. The winners will be the teams that understand tokens as an economic object and infrastructure as the mechanism that shapes that economy.

AI infrastructure is valuable because it is where scaling ambition meets physical constraint. As long as models keep wanting more compute and chips stop giving it away cheaply, the most important work will happen in the machinery between them.

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
[^gpu-price-performance]: Jaime Sevilla and Pablo Villalobos, [Trends in GPU Price-Performance](https://epoch.ai/publications/trends-in-gpu-price-performance), Epoch AI, 2022.
[^epoch-ai-trends]: Epoch AI, [Trends in Artificial Intelligence: AI Hardware](https://epoch.ai/trends), accessed 2026-07-02.
[^owid-gpu-price-performance]: Our World in Data, [GPU computational performance per dollar](https://ourworldindata.org/grapher/gpu-price-performance), accessed 2026-07-02.
