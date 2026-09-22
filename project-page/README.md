# CellAudit project page

Public research page: https://cellaudit-research.phuonganh49123.chatgpt.site

Code repository: https://github.com/limengran98/CellAudit

This is a self-contained static site. Serve `dist/` over HTTP:

```bash
python -m http.server 4173 --directory dist
```

Open http://localhost:4173. The page uses no build-time dependencies. `dist/assets/results.json` contains the numerical data behind the interactive displays, with manuscript-relative provenance. The figures reproduce the CellAudit paper; `dist/assets/CellAudit.pdf` is the named-author manuscript. Author, figure, PDF, and dataset rights remain with their respective owners.

The candidate explorer conditions on registered checkpoints and replacement maps. The feedback display uses paired-t 95% confidence intervals across five trajectories. The publication source and all page assets are mirrored in the public code repository under `project-page/`.
