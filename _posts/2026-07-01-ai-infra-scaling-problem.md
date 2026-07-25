---
layout: post
title: "[WIP] AI infra and tokenomics, part 2: why current AI infra is hard"
topic: "AI infrastructure"
sequence: 9
source_url: https://app.notion.com/p/38d2ec4bb1f0808ea061d11de43d93a6
source_label: "Original outline on Notion"
excerpt: "Why model size, request volume, context length, output length, and agent loops scale faster than hardware economics can hide."
---

This is the second part of the AI infrastructure and tokenomics series. [Part 1](/2026/06/30/ai-infra-and-tokenomics/) explained the historical shift from conventional request-serving infrastructure to agentic token-loop infrastructure. This part focuses on the pressure that makes the current layer so difficult: **the workload is scaling faster than hardware can hide.**

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

## The current problem

The hard part of current AI infrastructure is that demand scales along several axes at once:

- **Model size scaling:** more parameters, deeper networks, wider expert pools, larger activations.
- **User request scaling:** more users, more sessions, and more AI features embedded into products.
- **Context and output scaling:** longer prompts, larger KV cache, multimodal context, longer generated artifacts.
- **Agent-loop scaling:** one user can define a goal and let multiple agents run concurrently for hours.

At the same time, the hardware side is no longer a clean Dennard/Moore free lunch. Transistors still improve, but power, memory bandwidth, packaging, interconnect, and wafer cost increasingly decide whether peak math becomes useful work.

## Demand side: token work weak-scales

Scaling has multiple knobs:

- Model scale: parameters, activation size, depth, sparsity, experts.
- Training scale: tokens, data quality, optimizer budget, parallelism efficiency.
- Context scale: KV cache size, attention pattern, retrieval strategy.
- Inference scale: samples, reasoning steps, tool calls, verification passes.

The last knob is increasingly important. Brown et al. studied repeated sampling and found that solution coverage can continue improving as sample count grows over several orders of magnitude, with automatic verification converting samples into better measured performance in domains such as code and formal proofs.[^large-language-monkeys]

This changes the serving objective. A system may improve quality by spending more inference compute, but the product only wins if the extra compute is controlled. Good infra therefore does two things:

1. Lower the cost of each token.
2. Reduce the number of tokens required to complete the task.

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

## The infra engineering control loop

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

## Why this creates more room for infra programmers

In pre-LLM systems, infra optimization often improved a bounded request path: a faster feed ranking call, a cheaper video serving path, a more efficient cache, a better game backend. Those wins mattered, but the per-user unit of work was usually product-paced.

In LLM systems, infra optimization changes what the product can afford to attempt. A lower-cost attention kernel can make longer context feasible. Better KV-cache management can turn a product SLO from impossible to ordinary. Better batching and routing can make agentic workloads affordable. Better compiler lowering can decide which model shapes are practical at all.

That is the central problem of current AI infra: the middle layer has more complexity because both sides are moving. The algorithm side keeps asking for larger dynamic workloads, while the hardware side exposes increasingly specialized constraints. The opportunity is exactly the gap between them.

[Part 3](/2026/07/02/ai-infra-future/) looks at the supply-chain side of that gap: compute, memory, interconnect, edge accelerators, and why the future likely becomes more heterogeneous and more LLM-specific.

## References

[^openai-compute]: OpenAI, [AI and Compute](https://openai.com/index/ai-and-compute/), 2018.
[^epoch-compute]: Jaime Sevilla et al., [Compute Trends Across Three Eras of Machine Learning](https://arxiv.org/abs/2202.05924), 2022.
[^large-language-monkeys]: Bradley Brown et al., [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787), 2024.
[^kaplan]: Jared Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), 2020.
[^chinchilla]: Jordan Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), 2022.
[^sutter]: Herb Sutter, [The Free Lunch Is Over](https://www.cs.utexas.edu/~lin/cs380p/Free_Lunch.pdf), 2005.
[^dark-silicon]: Hadi Esmaeilzadeh et al., [Dark Silicon and the End of Multicore Scaling](https://research.cs.wisc.edu/vertical/papers/2011/isca11-darksilicon.pdf), 2011.
[^horowitz]: Mark Horowitz, [Computing's Energy Problem](https://gwern.net/doc/cs/hardware/2014-horowitz-2.pdf), ISSCC 2014.
