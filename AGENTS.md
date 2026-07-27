# Repository Guide

This is Kaiyu Shi's Jekyll-based technical blog. Favor clear, evidence-backed
technical writing about AI infrastructure, compilers, accelerators, and systems.

## Writing

- Treat new or `[WIP]` posts as reviewable drafts.
- Prefer a concise tech-blog structure: claim, system model, evidence, implications.
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
- Use precise SVG charts for quantitative data, axes, curves, and measured points.
- Every figure should use `<figure class="post-figure">`, descriptive alt text, and
  a caption that states the takeaway or methodology.
- Reuse a shared overview diagram across a series instead of creating near-duplicates.

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
- Check for stale asset references after renaming or deleting diagrams.
- Preview all affected posts locally and verify that every image loads, fits the
  article width, and remains readable without overlap.
- For published work, also verify the GitHub Pages build and live HTML/assets.
