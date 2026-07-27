---
layout: post
title: "[WIP] Why current AI infra is hard - AI-Infra Overview PART-2"
topic: "AI infrastructure"
sequence: 9
source_url: https://app.notion.com/p/38d2ec4bb1f0808ea061d11de43d93a6
source_label: "Original outline on Notion"
excerpt: "Why model size, request volume, context length, output length, and agent loops scale faster than hardware economics can hide."
---

This is the second part of the series. [Part 1](/2026/06/30/ai-infra-and-tokenomics/) followed the unit of work from a request to an agent loop. This part explains why that transition creates a qualitatively harder systems problem: **both the work inside one goal and the number of machine-generated steps behind it can grow.**

## The workload multiplies

The difference is not only that an LLM request is larger. It is that the user no longer has to pace every expensive operation:

<figure class="post-figure">
  <img src="{{ '/assets/user-agent-loop-excalidraw.svg' | relative_url }}" alt="Excalidraw comparison showing a traditional application waiting for the user after each bounded request, versus an agentic application where one delegated goal launches concurrent model, tool, and verification loops.">
  <figcaption>Traditional workloads return control to the user after each approximately bounded request; the system waits for the user to evaluate the response and make the next decision. For agentic workloads, \(W=A\cdot S\cdot T\cdot q\), where \(W\) is compute work per goal, \(A\) is concurrent agents, \(S\) is model or tool steps per agent, \(T\) is tokens per step, and \(q\) is compute per token. Total cost is \(C=U\cdot G\cdot W\cdot p/\eta\), where \(U\) is active users, \(G\) is goals per user, \(p\) is hardware price per unit compute, and \(\eta\) is full-stack efficiency. Infrastructure lowers cost by increasing \(\eta\) and by reducing the branches, steps, tokens, and data movement required to finish useful work.</figcaption>
</figure>

Pre-LLM consumer workloads usually keep most of these factors bounded. A person clicks, watches, scrolls, types, or plays; the user remains in the loop and paces expensive work. A video frame, feed request, or game tick can be optimized aggressively, but its shape does not expand because a model gained parameters or a context window doubled.

Agentic systems remove that pacing limit. One goal can launch several agents. Each agent can issue model calls, retrieve data, invoke tools, run tests, retry failures, and verify results for hours. At the same time, larger models, longer context, longer outputs, multimodal inputs, and inference-time reasoning increase \(T\) and \(q\). Brown et al. showed that repeated sampling with verification can improve measured solution coverage across large increases in sample count.[^large-language-monkeys]

This is the demand-side reason AI infrastructure matters. OpenAI's 2018 analysis found that compute used in the largest training runs had been doubling every 3.4 months from 2012 onward.[^openai-compute] Epoch AI's later estimate is more conservative but still aggressive: training compute for milestone systems has doubled roughly every 5 to 6 months since the early deep-learning era.[^epoch-compute] Kaplan et al. found power-law relationships between language-model loss and model size, data, and compute; Hoffmann et al. then showed that compute-optimal training requires model size and training tokens to scale together.[^kaplan][^chinchilla]

## Strong and weak scaling are the systems analogy

Strong and weak scaling have precise meanings in parallel computing:

<figure class="post-figure">
  <img src="{{ '/assets/strong-weak-scaling-canonical.svg' | relative_url }}" alt="Canonical comparison of strong scaling, where fixed total work finishes faster with more processors, and weak scaling, where work grows with processor count while time ideally remains constant.">
  <figcaption>Strong scaling fixes total work; weak scaling grows total work with resources. AI product demand increasingly resembles the second regime.</figcaption>
</figure>

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

Pre-LLM and post-LLM products are not literally parallel-computing benchmarks, but the analogy is useful. Conventional products mostly ask infrastructure to reduce the constant factor of a bounded, user-paced request. AI products repeatedly spend new hardware capacity on larger models, longer contexts, more reasoning, and more autonomous steps. The workload grows with the available system.

The bottleneck therefore moves instead of disappearing:

- More GPUs expose collective communication, topology, and straggler costs.
- Longer context turns KV-cache capacity and bandwidth into serving constraints.
- Larger batches improve throughput while increasing latency and memory pressure.
- MoE routing reduces active compute while adding placement and load-balancing problems.
- Test-time compute improves quality while making per-goal cost less predictable.

## Why every efficiency point is magnified

If an optimization leaves a fraction \(r\) of the original unit cost, then:

<div class="math-block">
$$
C_{\text{optimized}}=rC,
\qquad
\text{savings}=(1-r)C
$$
</div>

The percentage is ordinary; the base is not. As \(A\), \(S\), \(T\), and \(q\) grow, the same kernel, compiler, cache, quantization, batching, or routing improvement acts on more work. This is the economic meaning of "the more you buy, the more you save": not that scale makes waste acceptable, but that each percentage point of efficiency converts into a larger absolute saving.

Good infrastructure improves both sides of the equation:

1. Increase \(\eta\): execute each unit of work more efficiently.
2. Reduce \(W\): finish the goal with fewer tokens, tool calls, retries, or model passes.

## The AI infra control loop

Hardware no longer provides a uniform free lunch. Single-thread performance stopped scaling automatically; dark silicon made power limits explicit; and data movement can cost orders of magnitude more energy than arithmetic.[^sutter][^dark-silicon][^horowitz]

Modern accelerators answer with HBM, larger SRAM, tensor cores, lower precision, faster fabrics, and advanced packaging. Infrastructure turns those features into useful work through a recurring loop:

1. Measure the real workload: prefill/decode mix, cache residency, batch distribution, communication, and kernel hotspots.
2. Change model architecture or serving policy when the bottleneck is structural.
3. Change kernels, compiler lowering, layouts, and runtime scheduling when execution is the bottleneck.
4. Feed the remaining constraints back into model and hardware design.

That is why AI infrastructure has more room and more responsibility than the conventional middle layer. It can change not only how cheaply a product runs, but what the product can afford to attempt. [Part 3](/2026/07/02/ai-infra-future/) examines the supply side: compute, memory, communication, and edge hardware, and what their different trajectories imply for the future stack.

## References

[^openai-compute]: OpenAI, [AI and Compute](https://openai.com/index/ai-and-compute/), 2018.
[^epoch-compute]: Jaime Sevilla et al., [Compute Trends Across Three Eras of Machine Learning](https://arxiv.org/abs/2202.05924), 2022.
[^large-language-monkeys]: Bradley Brown et al., [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787), 2024.
[^kaplan]: Jared Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), 2020.
[^chinchilla]: Jordan Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), 2022.
[^sutter]: Herb Sutter, [The Free Lunch Is Over](https://www.cs.utexas.edu/~lin/cs380p/Free_Lunch.pdf), 2005.
[^dark-silicon]: Hadi Esmaeilzadeh et al., [Dark Silicon and the End of Multicore Scaling](https://research.cs.wisc.edu/vertical/papers/2011/isca11-darksilicon.pdf), 2011.
[^horowitz]: Mark Horowitz, [Computing's Energy Problem](https://gwern.net/doc/cs/hardware/2014-horowitz-2.pdf), ISSCC 2014.
