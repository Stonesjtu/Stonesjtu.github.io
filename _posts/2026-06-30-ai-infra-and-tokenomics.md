---
layout: post
title: "[WIP] From conventional infra to agentic infra - AI-Infra Overview PART-1"
topic: "AI infrastructure"
sequence: 8
source_url: https://app.notion.com/p/38d2ec4bb1f0808ea061d11de43d93a6
source_label: "Original outline on Notion"
excerpt: "A history of AI infrastructure as the implementation bridge between algorithmic workload demand and CPU/GPU/TPU/NPU hardware reality."
---

This is the first part of a three-part series on AI infrastructure and token economics. The full series is organized around one claim: **AI infrastructure is the implementation bridge between upstream workload design and underlying hardware reality.** Algorithms decide what work exists. CPUs, GPUs, TPUs, NPUs, memory, and networks decide what work is physically affordable. Infrastructure is the middle layer that makes the two meet.

The series:

1. **History:** from conventional infra to agentic infra, and why the unit of work changed.
2. **Problem:** [why model size, context, output length, and agent request volume scale faster than hardware economics](/2026/07/01/ai-infra-scaling-problem/).
3. **Future:** [why the next phase points toward heterogeneous computing and more dedicated LLM hardware](/2026/07/02/ai-infra-future/).

## Infra is the bridge

A useful way to look at AI infrastructure is not as "GPU operations" or "model serving" in isolation. It is the translation layer between workload semantics and hardware constraints:

```text
algorithm / workload intent -> infrastructure implementation -> hardware execution
```

The upstream side asks for larger models, longer context, multimodal inputs, retrieval, tool calls, verification, and background agents. The downstream side offers CPUs, GPUs, TPUs, NPUs, HBM, SRAM, PCIe, NVLink, Ethernet, InfiniBand, power budgets, and wafer economics. The infra layer decides how much useful work survives that translation.

That is why the token became a useful accounting unit. Training consumes tokens to create capability. Inference consumes tokens to deliver capability. Agentic workflows consume more tokens to plan, call tools, verify work, and recover from failure. Token cost is therefore not just a pricing metric. It is a full-stack performance metric.

<figure class="post-figure">
  <img src="{{ '/assets/ai-infra-token-stack.svg' | relative_url }}" alt="Stack diagram showing algorithm, model, serving, framework, compiler, kernel, accelerator, memory, network, chip design, and manufacturing all contributing to token economics.">
  <figcaption>The cost and latency of one useful token is the sum of many cross-layer decisions.</figcaption>
</figure>

## From conventional infra to agentic infra

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

## What changed

The historical shift is not only "more GPUs." It is a change in what the infrastructure system is asked to preserve:

- Conventional infrastructure preserves service availability and request latency.
- ML infrastructure preserves experiment reproducibility and accelerator allocation.
- Deep learning infrastructure preserves device utilization across training steps.
- LLM infrastructure preserves token throughput, KV-cache locality, and latency SLOs.
- Agentic infrastructure preserves useful task completion across long-running loops.

Each era makes the implementation bridge wider. More of the algorithm leaks into systems decisions, and more of the hardware leaks back into model design.

## Where the rest of the series goes

Part 2 turns this history into the current problem: AI workloads weak-scale because model size, token length, output length, and agent loops all grow. Part 3 looks at the supply side and argues that the future of AI infra is increasingly heterogeneous and LLM-specific.
