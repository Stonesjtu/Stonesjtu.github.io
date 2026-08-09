---
layout: post
title: "Why AI infra is fast-moving and challenging - AI-Infra Overview PART-2"
topic: "AI infrastructure"
sequence: 9
source_url: https://app.notion.com/p/38d2ec4bb1f0808ea061d11de43d93a6
source_label: "Original outline on Notion"
excerpt: "Why rapidly evolving models, request volume, context, output, agents, and hardware keep moving the AI infrastructure frontier."
---

This is the second part of the series. [Part 1](/2026/06/30/ai-infra-and-tokenomics/) showed how AI infrastructure maps evolving models onto evolving hardware. This part explains why that mapping is fast-moving and challenging: **the workload, hardware, and useful optimization frontier all change at once.** Both the work inside one goal and the number of machine-generated steps behind it can grow.

## Inference demand still outruns one accelerator

<p class="key-insight"><strong>Key insight</strong><span>The public full-context inference envelope has expanded much faster than the dense compute available from one accelerator.</span></p>

<figure class="post-figure post-chart">
  <section class="chart-panel">
    <p class="chart-title">Inference demand versus one accelerator</p>
    <p class="chart-subtitle">Normalized to BERT-Large request work and V100 compute in 2018; logarithmic scale</p>
    <div class="chart-frame chart-frame--tall"><canvas id="inference-demand-gap-chart" role="img" aria-label="Interactive logarithmic chart comparing public total model size, full-context inference work, and one accelerator's dense FP16 or BF16 compute from 2007 through 2026.">The normalized data is available in the table below.</canvas></div>
  </section>
  <figcaption>The chart normalizes workload and hardware to 2018. Violet uses total parameters; red uses \(P_{\text{active}}\times T_{\max}\), so sparse MoE models are not charged for every stored expert on every token. The 2026 GLM-5.2, DeepSeek-V4-Pro, and Kimi K3 points are horizontally jittered within the year for readability; GLM and Kimi active-parameter values are architecture-derived estimates. Hardware is one accelerator's dense FP16/BF16 peak, except the C870 FP32 legacy point; Rubin is preliminary. The open-model examples reach about 8,200x the BERT-Large parameter count and roughly 290,000x its full-context work envelope, versus about 32x in preliminary single-GPU compute.</figcaption>
</figure>

For a dense Transformer, a useful first-order proxy for one request is active parameters multiplied by processed tokens: \(W_{\text{request}}\propto P_{\text{active}}T\). This intentionally leaves out attention, KV-cache traffic, output decoding, repeated samples, and agent steps; all of those widen the real systems envelope.

Using public model specifications gives an estimated envelope. BERT-Large exposed 340 million parameters and 512-token sequences in 2018.[^bert] GPT-2 increased the model to 1.5 billion parameters in 2019, and GPT-3 reached 175 billion parameters with a 2,048-token context in 2020.[^gpt2][^gpt3] PaLM was a dense 540-billion-parameter model, while Llama 3.1 combined 405 billion parameters with a 128K context window.[^palm][^llama31]

The 2026 open-model frontier changes how that curve should be read. GLM-5.2 has a 753-billion-parameter checkpoint, DeepSeek-V4-Pro reports 1.6 trillion total parameters, and Kimi K3 reports 2.8 trillion.[^glm52][^deepseek-v4][^kimi-k3] All three are sparse MoE models with roughly 40–50 billion parameters active per token and about a one-million-token window. Total parameters therefore describe weight capacity, memory footprint, expert placement, and communication pressure; **active** parameters are the better first-order input to per-token arithmetic.

These are not average production requests; they show how much inference work a single request *can* ask the system to carry. The hardware curve uses NVIDIA's published dense FP16/BF16 figures, with C870 FP32 as a legacy proxy and Rubin marked preliminary.[^gpu-inference-trend]

<details class="post-details" markdown="1">
<summary>Show the normalized inference-demand assumptions</summary>

| public model example | year | total parameters | active parameters | advertised token window | total-parameter index | request-work index |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BERT-Large | 2018 | 340M | 340M | 512 | 1x | 1x |
| GPT-2 | 2019 | 1.5B | 1.5B | 1,024 | 4.4x | 8.8x |
| GPT-3 | 2020 | 175B | 175B | 2,048 | 515x | 2,059x |
| PaLM | 2022 | 540B | 540B | 2,048 | 1,588x | 6,353x |
| Llama 3.1 405B | 2024 | 405B | 405B | 128K | 1,191x | 297,794x |
| GLM-5.2 | 2026 | 753B | approximately 40B* | 1M | 2,215x | approximately 229,779x |
| DeepSeek-V4-Pro | 2026 | 1.6T | 49B | 1M | 4,706x | 281,480x |
| Kimi K3 | 2026 | 2.8T | approximately 50B* | 1M | 8,235x | approximately 287,224x |

The request-work index divides \(P_{\text{active}}T_{\max}\) by the BERT-Large baseline. It is a deliberately simple compute envelope, not a latency benchmark. Maximum context is not average context; prefill and decode have different execution shapes; and memory capacity, bandwidth, attention, batching, quantization, and KV-cache reuse determine how much of peak arithmetic becomes useful throughput. *GLM-5.2 and Kimi K3 do not publish an explicit active-parameter total in their launch material; the approximate values are derived from their disclosed MoE routing and checkpoint architecture, so they are shown as estimates rather than measurements.*

OpenAI publishes GPT-5.6 Sol's 1.05M-token context but not its parameter count.[^gpt56] A 2–4T total-parameter range is sometimes proposed as an external estimate; the chart shows that range as an unverified annotation and excludes it from both trend lines and index calculations.

</details>

This is the space AI infrastructure has to close. It increases effective hardware through lower precision, kernels, batching, parallelism, and communication overlap. It reduces demand through caching, sparsity, smaller routed models, speculative decoding, retrieval, and fewer wasted tokens or retries. The middle layer does not repeal the gap; it decides how much useful product can fit inside it.

## The workload multiplies

<p class="key-insight"><strong>Key insight</strong><span>A traditional application pauses for the next human decision; an agentic application can turn one delegated goal into many machine-paced branches, steps, and retries.</span></p>

<figure class="post-figure">
  <img src="{{ '/assets/user-agent-loop-excalidraw.svg' | relative_url }}?v=20260809b" alt="Excalidraw comparison showing one traditional request returning a result and waiting for another user decision, while one delegated agent goal launches several concurrent model, tool, and verification loops before producing completed work.">
  <figcaption>Traditional workloads return control to the user after each approximately bounded request; the system waits for the user to evaluate the response and make the next decision. For agentic workloads, \(W=A\cdot S\cdot T\cdot q\), where \(W\) is compute work per goal, \(A\) is concurrent agents, \(S\) is model or tool steps per agent, \(T\) is tokens per step, and \(q\) is compute per token. Total cost is \(C=U\cdot G\cdot W\cdot p/\eta\), where \(U\) is active users, \(G\) is goals per user, \(p\) is hardware price per unit compute, and \(\eta\) is full-stack efficiency. Infrastructure lowers cost by increasing \(\eta\) and by reducing the branches, steps, tokens, and data movement required to finish useful work.</figcaption>
</figure>

Pre-LLM consumer workloads usually keep most of these factors bounded. A person clicks, watches, scrolls, types, or plays; the user remains in the loop and paces expensive work. A video frame, feed request, or game tick can be optimized aggressively, but its shape does not expand because a model gained parameters or a context window doubled.

Agentic systems remove that pacing limit. One goal can launch several agents. Each agent can issue model calls, retrieve data, invoke tools, run tests, retry failures, and verify results for hours. At the same time, larger models, longer context, longer outputs, multimodal inputs, and inference-time reasoning increase \(T\) and \(q\). Brown et al. showed that repeated sampling with verification can improve measured solution coverage across large increases in sample count.[^large-language-monkeys]

## Strong and weak scaling are the systems analogy

<p class="key-insight"><strong>Key insight</strong><span>Traditional infrastructure is usually asked to finish bounded, human-triggered work faster; agentic products spend added capacity on more work inside each goal.</span></p>

<figure class="post-figure post-chart">
  <div class="chart-grid chart-grid--two">
    <section class="chart-panel">
      <p class="chart-title">Strong scaling</p>
      <p class="chart-subtitle">Fixed total work; communication creates a floor</p>
      <div class="chart-frame chart-frame--compact"><canvas id="strong-scaling-chart" role="img" aria-label="Interactive chart showing normalized completion time falling as processor count increases for a fixed workload.">The relationship is described by the formula below.</canvas></div>
    </section>
    <section class="chart-panel">
      <p class="chart-title">Weak scaling</p>
      <p class="chart-subtitle">Fixed work per processor; total work grows with resources</p>
      <div class="chart-frame chart-frame--compact"><canvas id="weak-scaling-chart" role="img" aria-label="Interactive chart showing total workload growing with processor count while work per processor remains constant.">The relationship is described by the formula below.</canvas></div>
    </section>
  </div>
  <figcaption>Strong scaling fixes total work; weak scaling grows total work with resources. AI product demand increasingly resembles the second regime.</figcaption>
</figure>

The analogy becomes clearer when comparing the control planes. Traditional Dev Infra and ML Infra automate large jobs, but humans usually decide when a build, test, deployment, or experiment begins. Agentic Infra also automates the creation of *new work inside the goal*: the next model call, tool action, retry, verification pass, or concurrent subtask.

| dimension | traditional Dev / ML Infra | current Agentic Infra |
| --- | --- | --- |
| trigger | engineer submits a build, test, deployment, or experiment | user delegates a goal once |
| unit of work | bounded pipeline or predefined job graph | evolving task tree of model calls, tools, and checks |
| pacing loop | system finishes and waits for human feedback | machine decides and launches the next step |
| work per trigger | mostly fixed after submission | compounds across agents, steps, tokens, retries, and tools |
| concurrency | bounded by teams, commits, and planned experiments | many agents and branches can run for one user concurrently |
| natural limiter | human attention and decision latency | compute budget, policy, quality threshold, and infrastructure capacity |
| optimization target | turnaround time, reliability, reproducibility, utilization | useful completed work per dollar, watt, second, and user goal |

Traditional Dev/ML demand can still grow substantially, and one training experiment can be enormous. The important bound is the request-generation loop: new jobs usually enter at human speed. Agentic demand can grow much faster because one human decision creates a machine-paced loop. Its work envelope multiplies as \(A\cdot S\cdot T\cdot q\), so increases in autonomy, model size, context, reasoning, and concurrency can produce an exponential-looking demand curve without requiring exponentially more users.

Strong and weak scaling have precise meanings in parallel computing.

For strong scaling, total work \(W\) is fixed and more parallel resources \(P\) reduce completion time:

<div class="math-block">
$$
T_{\text{strong}}(W,P)
\approx
\frac{W}{P}+T_{\text{comm}}(P)
$$
</div>

For weak scaling, work grows with resources so that work per processor stays approximately constant:

<div class="math-block">
$$
W(P) \approx P \cdot W_0,
\qquad
T_{\text{weak}}(P)
\approx
W_0+T_{\text{comm}}(P)
$$
</div>

Dev/ML and agentic systems are not literally parallel-computing benchmarks, but the analogy is useful. Traditional infrastructure mostly asks the system to finish a bounded, human-submitted job faster: a strong-scaling-shaped objective. Agentic products repeatedly spend new capacity on larger models, longer contexts, more reasoning, more branches, and more autonomous steps. The workload expands with the available system, which makes it weak-scaling-shaped.

The bottleneck therefore moves instead of disappearing:

- More GPUs expose collective communication, topology, and straggler costs.
- Longer context turns KV-cache capacity and bandwidth into serving constraints.
- Larger batches improve throughput while increasing latency and memory pressure.
- MoE routing reduces active compute while adding placement and load-balancing problems.
- Test-time compute improves quality while making per-goal cost less predictable.

## Why every efficiency point is magnified

<p class="key-insight"><strong>Key insight</strong><span>A fixed percentage improvement saves more absolute money as agents multiply branches, steps, tokens, and compute per token.</span></p>

<div class="math-block">
$$
C_{\text{optimized}}=rC,
\qquad
\text{savings}=(1-r)C
$$
</div>

If an optimization leaves a fraction \(r\) of the original unit cost, the percentage is ordinary; the multiplying workload base is not.

As \(A\), \(S\), \(T\), and \(q\) grow, the same kernel, compiler, cache, quantization, batching, or routing improvement acts on more work. This is the economic meaning of "the more you buy, the more you save": not that scale makes waste acceptable, but that each percentage point of efficiency converts into a larger absolute saving.

Good infrastructure improves both sides of the equation:

1. Increase \(\eta\): execute each unit of work more efficiently.
2. Reduce \(W\): finish the goal with fewer tokens, tool calls, retries, or model passes.

## The AI infra control loop

<p class="key-insight"><strong>Key insight</strong><span>AI infrastructure repeatedly measures the workload, finds the binding hardware constraint, and changes models, kernels, compilers, or scheduling to recover useful capacity.</span></p>

Hardware no longer provides a uniform free lunch. Single-thread performance stopped scaling automatically; dark silicon made power limits explicit; and data movement can cost orders of magnitude more energy than arithmetic.[^sutter][^dark-silicon][^horowitz]

Modern accelerators answer with HBM, larger SRAM, tensor cores, lower precision, faster fabrics, and advanced packaging. Infrastructure turns those features into useful work through a recurring loop:

1. Measure the real workload: prefill/decode mix, cache residency, batch distribution, communication, and kernel hotspots.
2. Change model architecture or serving policy when the bottleneck is structural.
3. Change kernels, compiler lowering, layouts, and runtime scheduling when execution is the bottleneck.
4. Feed the remaining constraints back into model and hardware design.

That is why AI infrastructure has more room and more responsibility than the conventional middle layer. It can change not only how cheaply a product runs, but what the product can afford to attempt. [Part 3](/2026/07/02/ai-infra-future/) examines the supply side: compute, memory, communication, and edge hardware, and what their different trajectories imply for the future stack.

## References

[^large-language-monkeys]: Bradley Brown et al., [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787), 2024.
[^bert]: Jacob Devlin et al., [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805), 2018; Google Research, [BERT model repository](https://github.com/google-research/bert), documenting the 340M-parameter BERT-Large model and 512-token sequence length.
[^gpt2]: OpenAI, [Better Language Models and Their Implications](https://openai.com/index/better-language-models/), 2019; Alec Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), documenting GPT-2's 1.5B parameters and 1,024-token context.
[^gpt3]: Tom Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), 2020, documenting 175B parameters and a 2,048-token context.
[^palm]: Aakanksha Chowdhery et al., [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311), 2022; Reiner Pope et al., [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102), documenting dense 540B-parameter inference with a 2,048-token context.
[^llama31]: Meta, [Introducing Llama 3.1](https://ai.meta.com/blog/meta-llama-3-1/), 2024, documenting the 405B model and 128K context window.
[^glm52]: Z.ai, [GLM-5.2: Built for Long-Horizon Tasks](https://z.ai/blog/glm-5.2), 2026, documenting the 1M-token context and sparse-attention architecture; the official [GLM-5.2 Hugging Face repository](https://huggingface.co/zai-org/GLM-5.2) reports a 753B-parameter checkpoint, while its [configuration](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json) exposes 256 routed experts with eight selected per token. The approximately 40B active estimate uses the disclosed architecture and the [GLM-5 technical report](https://arxiv.org/abs/2602.15763), which reports 40B active parameters for the closely related 744B checkpoint.
[^deepseek-v4]: DeepSeek, [DeepSeek V4 Preview Release](https://api-docs.deepseek.com/news/news260424/), 2026, documenting 1.6T total and 49B active parameters for V4-Pro with a 1M-token context.
[^kimi-k3]: Moonshot AI, [Kimi K3 Tech Blog](https://www.kimi.com/blog/kimi-k3), 2026, documenting 2.8T total parameters, a 1M-token context, and 16 selected experts among 896. The approximately 50B active value is an architecture-derived estimate rather than an official published total.
[^gpt56]: OpenAI, [GPT-5.6 Sol model documentation](https://developers.openai.com/api/docs/models/gpt-5.6-sol), 2026, documenting a 1.05M-token context window. OpenAI's release material does not disclose total or active parameter counts.
[^gpu-inference-trend]: NVIDIA, [Tesla technical brief](https://www.nvidia.com/docs/io/43395/tesla_technical_brief.pdf), [P100 architecture](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf), [V100 architecture](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf), [A100 architecture](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf), [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/), [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/), and preliminary [Vera Rubin NVL72 specifications](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/). The normalized hardware series uses 0.518, 21.2, 125, 312, 989, 2,250, and preliminary 4,000 TFLOP/s for C870, P100, V100, A100, H100, B200, and Rubin respectively; C870 uses FP32 as a pre-native-FP16 proxy.
[^sutter]: Herb Sutter, [The Free Lunch Is Over](https://www.cs.utexas.edu/~lin/cs380p/Free_Lunch.pdf), 2005.
[^dark-silicon]: Hadi Esmaeilzadeh et al., [Dark Silicon and the End of Multicore Scaling](https://research.cs.wisc.edu/vertical/papers/2011/isca11-darksilicon.pdf), 2011.
[^horowitz]: Mark Horowitz, [Computing's Energy Problem](https://gwern.net/doc/cs/hardware/2014-horowitz-2.pdf), ISSCC 2014.

<script defer src="{{ '/assets/vendor/chart.umd.min.js' | relative_url }}"></script>
<script defer src="{{ '/assets/chart-theme.js' | relative_url }}"></script>
<script defer src="{{ '/assets/series-charts.js' | relative_url }}?v=20260809f"></script>
