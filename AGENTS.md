# Repository Guide

This is Kaiyu Shi's Jekyll-based technical blog. Favor clear, evidence-backed
technical writing about AI infrastructure, compilers, accelerators, and systems.

## Writing

- Treat new or `[WIP]` posts as reviewable drafts.
- Prefer a concise tech-blog structure: claim, system model, evidence, implications.
- Within a major section, prefer this reading order: title, one-sentence key insight,
  diagram/plot/formula/table, then detailed explanation, methodology, and caveats.
  Let readers see the argument before asking them to parse its derivation.
- Frame AI infrastructure around one durable job: run a model on hardware under a
  service objective. Explain the model execution pattern, the hardware constraints,
  and the infrastructure mapping between them.
- When describing infrastructure evolution, follow the causal chain: model behavior
  -> hardware bottleneck -> infrastructure technique -> service outcome. Scheduling
  is one technique, not the definition of an infrastructure era.
- Distinguish training, inference, and agentic execution when their compute, memory,
  communication, state, or reliability requirements differ.
- Remove repetition before adding more text.
- Keep terminology and metrics consistent across prose, tables, captions, and charts.
- Use neutral contribution wording such as "worked on" or "contributed to" unless
  the author explicitly claims ownership or leadership.
- Use MathJax for formulas and explain every symbol near its first use.
- Put long datasets, methodology, and secondary detail in `<details>` blocks.

## Evidence

- Use primary sources where possible: vendor specifications, papers, standards,
  launch material, and official pricing.
- Clearly label estimates, proxies, mixed-precision comparisons, and forecast data.
- Cite claims close to the supporting text and define chart methodology in the
  caption or nearby prose.
- When changing a metric, update the data, axis direction, title, caption, alt text,
  and surrounding argument together.

## Diagrams

- Use Excalidraw for conceptual flows, timelines, system maps, and topology diagrams.
- Store both the editable `.excalidraw` scene and its exported `.svg` in `assets/`.
- Use Chart.js with the shared repository theme for quantitative data, axes, curves,
  and measured points. Keep static SVG charts only when the surrounding post already
  relies on that format or a script-free figure is materially better.
- For AI-infrastructure evolution diagrams, align three layers explicitly: model
  execution, infrastructure mapping, and hardware substrate. Make the infrastructure
  layer visually central and show the mapping direction between the three.
- Give conceptual diagrams one primary reading direction. Timelines should use
  consistent eras, and captions must say when eras overlap rather than implying
  precise replacement dates.
- Put short concepts in boxes and move evidence, qualifications, symbols, and method
  detail into the caption or prose. Split a diagram when labels become too small or
  when it tries to make more than one major argument.
- Use the causal relationship as the visual grammar: a model property creates a
  resource pressure, infrastructure applies a mechanism, and hardware executes it.
  Do not organize a history only as a list of scheduler units or product names.
- Every figure should use `<figure class="post-figure">`, descriptive alt text, and
  a caption that states the takeaway or methodology.
- Reuse a shared overview diagram across a series instead of creating near-duplicates.
- Keep diagram terminology synchronized with the surrounding headings and prose.
  Historical milestone labels should be backed by primary sources nearby.

## Editing

- Follow existing Jekyll front matter and `{{ '/assets/...' | relative_url }}` paths.
- Keep changes scoped and do not discard unrelated work in a dirty worktree.
- Do not commit, push, or publish unless explicitly requested.
- Avoid introducing new build dependencies when the existing static structure works.

## Validation

- Run `git diff --check`.
- Validate edited SVGs with `xmllint --noout`.
- Parse edited `.excalidraw` files as JSON and confirm exported SVGs contain
  `svg-source:excalidraw`.
- Run `node --check` on edited chart scripts and verify Chart.js canvases render
  without console errors.
- Check for stale asset references after renaming or deleting diagrams.
- Preview all affected posts locally and verify that every image loads, fits the
  article width, and remains readable without overlap. Check a normal desktop width
  and a narrow mobile width; shorten, split, or re-layout figures instead of shrinking
  explanatory text into illegibility.
- For published work, also verify the GitHub Pages build and live HTML/assets.
