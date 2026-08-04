---
layout: post
title: "[WIP] From conventional infra to agentic infra - AI-Infra Overview PART-1"
topic: "AI infrastructure"
sequence: 8
source_url: https://app.notion.com/p/38d2ec4bb1f0808ea061d11de43d93a6
source_label: "Original outline on Notion"
excerpt: "A history of AI infrastructure as the implementation bridge between algorithmic workload demand and CPU/GPU/TPU/NPU hardware reality."
---

This is the first part of a four-part series organized around one claim: **AI infrastructure is the implementation bridge between upstream workload design and underlying hardware reality.**

The series:

1. **History:** how the unit of work changed from a request into an agent loop.
2. **Problem:** [why model size, context, output length, and agent request volume scale faster than hardware economics](/2026/07/01/ai-infra-scaling-problem/).
3. **Future:** [why the next phase points toward heterogeneous computing and more dedicated LLM hardware](/2026/07/02/ai-infra-future/).
4. **Edge:** [how local hardware economics and small-model progress determine the intelligence that can stay on-device](/2026/07/03/ai-infra-edge-intelligence/).

## Infra is the bridge

A model describes *what* computation should happen. Hardware determines *what is physically and economically possible*. Infrastructure translates between them: it chooses model placement, numerical formats, batching, caching, parallelism, kernels, communication, and failure handling.

<figure class="post-figure">
  <img src="{{ '/assets/ai-infra-unified-map.svg' | relative_url }}" alt="Excalidraw systems map showing workload pressures flowing through AI infrastructure placement, compilation, caching, scheduling, runtime, and data movement into compute, memory, communication, and edge hardware constraints, with a feedback loop.">
  <figcaption>This is the backbone of the series: infrastructure turns workload intent into useful execution, while hardware constraints feed back into model and system design.</figcaption>
</figure>

The upstream side now asks for larger models, longer context, multimodal inputs, retrieval, tool calls, verification, and background agents. The downstream side offers increasingly specialized CPUs, GPUs, TPUs, NPUs, SRAM, HBM, fabrics, and packaging. Neither side can be designed independently: model choices leak into systems decisions, and hardware constraints leak back into model design.

Tokens are useful for accounting because they cross this boundary. Training spends tokens to create capability. Inference spends tokens to deliver it. Agentic workflows spend additional tokens to plan, call tools, verify results, and recover from failure. The real objective is not the cheapest token in isolation; it is the most useful completed work per dollar, watt, and second.

## From conventional infra to agentic infra

<figure class="post-figure">
  <img src="{{ '/assets/infra-evolution-agentic-excalidraw.svg' | relative_url }}" alt="Excalidraw timeline showing infrastructure evolving from conventional requests through machine-learning experiments, deep-learning training steps, LLM tokens and KV cache, and agentic task loops.">
  <figcaption>The important shift is the scheduled unit of work: request, experiment, training step, token and KV state, then a goal-driven task loop. Each transition adds autonomy, state, and accelerator intensity.</figcaption>
</figure>

The scheduler reveals what each generation values. Conventional infrastructure schedules requests and preserves availability. ML infrastructure schedules experiments and preserves reproducibility. Deep-learning infrastructure schedules kernel graphs and keeps accelerators fed. LLM infrastructure schedules tokens and KV-cache blocks while balancing throughput against latency.

Agentic infrastructure schedules something more stateful: a task loop. One user goal can expand into planning, retrieval, model calls, tool execution, tests, retries, and verification. The system may run for hours without the user pacing every step. Its performance target therefore moves beyond tokens per second toward **useful completed work per token**.

That change creates the central problem for [Part 2](/2026/07/01/ai-infra-scaling-problem/): both the work inside one goal and the number of machine-generated requests behind that goal can grow. [Part 3](/2026/07/02/ai-infra-future/) then asks what hardware and infrastructure become necessary when compute, memory, communication, and edge constraints all evolve at different rates.
