# DL261 Assignments

Public course-project repository for **CO3133 - Deep Learning and Its Applications**, Semester-261.

This repository hosts the GitHub Pages site and assignment deliverables for **CO3133 - Deep Learning and Its Applications**, Semester-261.

## GitHub Pages

- Landing page: `docs/index.html`
- Assignment 1: `docs/assignments/a1.html`
- Assignment 2: `docs/assignments/a2.html`
- Assignment 3: `docs/assignments/a3.html`

Live Pages URL: <https://LongQuach45.github.io/DL261-Assignments/>

## Repository Layout

```text
.
├── AI_USAGE.md
├── README.md
├── assignments/
│   ├── assignment-1/
│   │   ├── source/
│   │   ├── report/
│   │   ├── slide/
│   │   └── video/
│   ├── assignment-2/
│   │   ├── source/
│   │   ├── report/
│   │   ├── slide/
│   │   └── video/
│   └── assignment-3/
│       ├── source/
│       ├── report/
│       ├── slide/
│       └── video/
├── config/
│   └── deliverables.yml
├── checkpoints/
├── docs/
│   ├── index.html
│   ├── assets/
│   │   └── styles.css
│   └── assignments/
│       ├── a1.html
│       ├── a2.html
│       └── a3.html
```

## Installation

Assignment 1 currently has a partial source implementation. Install its dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r assignments/assignment-1/source/requirements.txt
```

## Dataset Preparation

TODO: Add dataset download, storage paths, licenses, preprocessing commands, split seeds, and leakage-prevention rules for each assignment.

## Training

TODO: Add reproducible training commands for each assignment.

Example placeholder:

```bash
cd assignments/assignment-1/source
python main.py
```

## Evaluation

TODO: Add evaluation commands, checkpoint paths, expected metrics, and links to logs or experiment IDs.

Example placeholder:

```bash
TODO: Add evaluation command after the evaluation script is implemented.
```

## Reproducibility Checklist

- TODO: Configuration files for every reported run.
- TODO: Fixed seed settings.
- TODO: Dependency versions.
- TODO: Hardware information.
- TODO: Checkpoint access or reconstruction instructions.
- TODO: Links to source, reports, slides, videos, and assignment pages.
- TODO: Commit/tag mapping for final reported results.

## AI Usage

See [AI_USAGE.md](AI_USAGE.md). Complete the disclosure before submission and keep it consistent with the landing page, assignment pages, reports, slides, and videos.
