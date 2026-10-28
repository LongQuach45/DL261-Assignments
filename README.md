# DL261 Assignments — MobyDick

Public course-project repository for **CO3133: Deep Learning and Its Applications**, Semester-261, Faculty of Computer Science and Engineering, Ho Chi Minh City University of Technology – VNU-HCM.

Live website: <https://longquach45.github.io/DL261-Assignments/>

## Website

The GitHub Pages site is an Astro static project with Tailwind CSS. It presents the required course, group, assignment, deliverable, and AI-use information while importing verified experiment figures and metrics from the repository.

```bash
npm ci
npm run dev
npm run verify
```

`npm run verify` performs Astro type checking, creates the production build, and checks handbook-required page content. GitHub Actions deploys `dist/` to GitHub Pages after a push to `main`.

## Project Layout

```text
.
├── assignments/
│   ├── assignment-1/     # Fashion-MNIST architecture comparison
│   ├── assignment-2/     # Food-101 specialized image classification
│   └── assignment-3/     # Multimodal deep learning, planned
├── config/deliverables.yml
├── src/                  # Astro pages, components, metadata, and styles
├── scripts/check-handbook.mjs
├── .github/workflows/    # GitHub Pages deployment
└── AI_USAGE.md
```

## Assignment Reproduction

Each completed assignment contains its own installation, data preparation, training, evaluation, checkpoint, environment, and result-traceability instructions:

- [Assignment 1 source README](assignments/assignment-1/source/README.md)
- [Assignment 2 source README](assignments/assignment-2/source/README.md)
- [Assignment 2 Dataset Proposal](assignments/assignment-2/proposal/dataset.md)

## AI Usage

See [AI_USAGE.md](AI_USAGE.md) for course-level disclosure and the assignment source folders for the detailed records.
