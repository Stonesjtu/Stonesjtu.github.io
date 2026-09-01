---
layout: post
title: "From conventional infra to agentic infra - AI-Infra Overview PART-1"
topic: "AI infrastructure"
sequence: 8
last_modified_at: 2026-08-09T23:40:14+08:00
source_url: https://app.notion.com/p/38d2ec4bb1f0808ea061d11de43d93a6
source_label: "Original outline on Notion"
excerpt: "A history of AI infrastructure through the co-evolution of model execution patterns and CPU/GPU/TPU/NPU hardware."
---

This is the first part of a four-part series organized around one claim: **AI infrastructure is the implementation bridge between upstream workload design and underlying hardware reality.**

The series:

1. **History:** how model execution and hardware specialization co-evolved, and how infrastructure kept them connected.
2. **Problem:** [why model size, context, output length, and agent request volume scale faster than hardware economics](/2026/07/01/ai-infra-scaling-problem/).
3. **Future:** [why the next phase points toward heterogeneous computing and more dedicated LLM hardware](/2026/07/02/ai-infra-future/).
4. **Edge:** [how local hardware economics and small-model progress determine the intelligence that can stay on-device](/2026/07/03/ai-infra-edge-intelligence/).

## Infra is the bridge

<p class="key-insight"><strong>Key insight</strong><span>AI infrastructure implements a model on hardware under a service objective.</span></p>

<figure class="post-figure">
  <img src="{{ '/assets/ai-infra-unified-map.svg' | relative_url }}?v=20260809" alt="Excalidraw framework titled AI Infra connects Model to Hardware. The model defines computation, state, and execution mode; AI infrastructure compiles, partitions, places, moves, caches, schedules, and operates it; hardware supplies compute engines, memory hierarchy, topology, and power; a service objective evaluates quality, latency, throughput, cost, energy, and reliability.">
  <figcaption>Read left to right. The model defines computation, state, and execution mode. Hardware supplies compute engines, memory hierarchy, topology, and power. AI infrastructure is the compiler, runtime, and distributed system that maps between them. Every mapping decision is judged against the service objective at the bottom.</figcaption>
</figure>

AI infrastructure is neither the model nor the silicon. It is the implementation layer that makes a model execute efficiently and reliably on a physical machine.

This diagram gives a compact review framework for any AI-infrastructure technique:

1. **Model contract:** which operators, tensor shapes, numerical formats, parameters, activations, context, or agent state must execute?
2. **Hardware budget:** which compute engines, memory levels, links, storage, power, and placement constraints are available?
3. **Infrastructure mapping:** how should the system compile, partition, place, move, cache, batch, schedule, isolate, and recover that work?
4. **Service objective:** did the mapping improve useful quality, latency, throughput, cost, energy, or reliability?

Model and hardware design still feed back into each other. A memory limit can motivate quantization, sparsity, or a smaller active parameter set; a model's attention or routing pattern can motivate new kernels, cache layouts, fabrics, or accelerators. Tokens remain useful for accounting, but they are not the goal by themselves. The real target is useful model execution per dollar, watt, and second, within the required quality and reliability envelope.

## From conventional infra to agentic infra

<p class="key-insight"><strong>Key insight</strong><span>Across four eras, AI infrastructure kept the same job while changing its mapping strategy as models and hardware co-evolved.</span></p>

<figure class="post-figure">
  <img src="{{ '/assets/infra-evolution-agentic-excalidraw.svg' | relative_url }}" alt="Excalidraw timeline with model, infrastructure, and hardware lanes showing compact machine-learning models on CPUs evolving into deep neural networks on GPUs and TPUs, transformers on HBM-connected accelerator clusters, and multimodal agent loops on heterogeneous systems, with Google Translate, Google Photos and neural Translate, GPT-3 API and ChatGPT, and OpenAI Codex as representative products.">
  <figcaption>Read each column vertically: a model's execution pattern creates a mapping problem for infrastructure, which must fit that work onto the available hardware. Gold tags anchor each era to representative products rather than claiming an exclusive mapping. The eras overlap rather than switch on exact dates; the timeline highlights the dominant new pressure in each period.</figcaption>
</figure>

AI infrastructure has one durable job: **run a model on hardware under a service objective**. The model supplies an execution graph and state; the hardware supplies compute units, memory hierarchy, interconnect, storage, and power. Infrastructure decides how to map one onto the other without violating latency, throughput, reliability, or cost targets.

### Compact models fit commodity infrastructure

In the CPU-centric era, production models were commonly compact classifiers, rankers, trees, or manually assembled feature pipelines. Their inference usually fit inside a conventional request path: load features from DRAM, run bounded computation on a CPU, and return a score. Infrastructure work concentrated on data pipelines, feature consistency, request scheduling, replication, and availability. Google Translate is a useful product anchor: its 2006 launch used statistical machine learning and large language-statistics pipelines rather than an end-to-end neural model.[^translate-history] The model was important, but it rarely dictated the shape of the datacenter.

### Dense neural networks made accelerators part of the system

Deep learning changed the hardware contract. AlexNet paired a 60-million-parameter convolutional network with a GPU implementation and split the model across two GTX 580 GPUs because device memory was a binding constraint.[^alexnet] Dense tensor operations rewarded massively parallel processors; device memory, host-to-device copies, kernel launch overhead, and gradient synchronization became first-class systems concerns. Distributed training frameworks and accelerator-aware runtimes emerged to keep GPUs fed and to coordinate replicas.

Hardware then began specializing for the model. Google's first production TPU, deployed in 2015, centered a 65,536-element 8-bit matrix multiply unit and 28 MiB of on-chip memory around neural-network inference.[^tpu] User-facing products made the shift visible: Google Photos launched in 2015 with automatic organization and search, while Google Translate moved eight language pairs to Neural Machine Translation in 2016.[^deep-products] This is the recurring pattern: once a model family exposes a stable and valuable computation structure, infrastructure and silicon reorganize around it.

### Transformers turned memory and communication into the bottleneck

The Transformer replaced recurrence with attention, making training more parallel while tying work and memory traffic to sequence length.[^transformer] By GPT-3, a single autoregressive model had reached 175 billion parameters,[^gpt3] far beyond one accelerator's memory. Infrastructure had to shard weights and activations across devices, combine data, tensor, and pipeline parallelism, and treat the interconnect as part of the computer.

Inference added a different execution pattern: **prefill** processes the prompt in parallel, while **decode** repeatedly generates one token and reads a growing KV cache. The systems response therefore moved beyond generic GPU scheduling. FlashAttention reduces traffic between HBM and on-chip SRAM,[^flashattention] while PagedAttention applies virtual-memory-style paging to dynamic KV-cache allocation.[^pagedattention] These are infrastructure techniques derived directly from the model's memory behavior and the hardware's hierarchy.

The product transition was equally legible. The GPT-3-backed OpenAI API exposed a general-purpose text interface in 2020, and ChatGPT turned a GPT-3.5-series model into a conversational product in November 2022.[^foundation-products] Both depended on a serving layer that hid distributed model execution behind a simple request interface.

### Agents make model execution a stateful workflow

Agentic systems do not replace the model-hardware mapping; they wrap it in a larger loop. ReAct demonstrated the core pattern by interleaving model-generated reasoning with actions against external environments.[^react] OpenAI Codex is a representative product: one delegated software task runs in its own cloud environment, where the agent can inspect a repository, edit code, execute tests, and propose a pull request.[^codex-product] A production agent generalizes that loop with retrieval, tool calls, code execution, tests, retries, and durable state. Different steps may belong on different resources: large-model decoding on GPUs or TPUs, embeddings or small models on NPUs, control logic on CPUs, and working memory in caches, databases, or object storage.

The infrastructure unit is therefore no longer only a request, experiment, training step, or token. It is a **stateful execution graph whose model calls must be placed on heterogeneous hardware**. Scheduling still matters, but it is one mechanism inside a broader mapping problem that includes compilation, numerical precision, parallelism, cache placement, communication, recovery, and tool isolation. The service objective also expands from tokens per second toward **useful completed work per dollar, watt, and second**.

That change creates the central problem for [Part 2](/2026/07/01/ai-infra-scaling-problem/): both the work inside one goal and the number of machine-generated requests behind that goal can grow. [Part 3](/2026/07/02/ai-infra-future/) then asks what hardware and infrastructure become necessary when compute, memory, communication, and edge constraints all evolve at different rates.

[^alexnet]: Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, ["ImageNet Classification with Deep Convolutional Neural Networks"](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), NeurIPS 2012. The paper describes the 60-million-parameter network, its GPU implementation, and the split across two 3 GB GTX 580 GPUs.
[^tpu]: Norman P. Jouppi et al., ["In-Datacenter Performance Analysis of a Tensor Processing Unit"](https://arxiv.org/abs/1704.04760), ISCA 2017.
[^transformer]: Ashish Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), NeurIPS 2017.
[^gpt3]: Tom B. Brown et al., ["Language Models are Few-Shot Learners"](https://arxiv.org/abs/2005.14165), NeurIPS 2020.
[^flashattention]: Tri Dao et al., ["FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"](https://arxiv.org/abs/2205.14135), NeurIPS 2022.
[^pagedattention]: Woosuk Kwon et al., ["Efficient Memory Management for Large Language Model Serving with PagedAttention"](https://arxiv.org/abs/2309.06180), SOSP 2023.
[^react]: Shunyu Yao et al., ["ReAct: Synergizing Reasoning and Acting in Language Models"](https://arxiv.org/abs/2210.03629), ICLR 2023.
[^translate-history]: Google, ["20 fun facts to celebrate Google Translate turning 20"](https://blog.google/products-and-platforms/products/translate/fun-facts-google-translate-20-years/), 2026.
[^deep-products]: Google, ["Picture this: A fresh approach to Photos"](https://blog.google/products-and-platforms/products/photos/picture-this-fresh-approach-to-photos/), 2015; Google, ["Found in translation: More accurate, fluent sentences in Google Translate"](https://blog.google/products-and-platforms/products/translate/found-translation-more-accurate-fluent-sentences-google-translate/), 2016.
[^foundation-products]: OpenAI, ["OpenAI API"](https://openai.com/index/openai-api/), 2020; OpenAI, ["Introducing ChatGPT"](https://openai.com/index/chatgpt/), 2022.
[^codex-product]: OpenAI, ["Introducing Codex"](https://openai.com/index/introducing-codex/), 2025.
