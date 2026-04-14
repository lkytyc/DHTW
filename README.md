# DHTW: Domain-Specific Semantic-Rich Knowledge Graph Construction through Human-LLM Team Working

## Overview

DHTW is a research-oriented repository for constructing domain-specific,
semantic-rich knowledge graphs with coordinated human and LLM participation.
The project contains:

- a main DHTW pipeline for knowledge-graph exploration and construction
- prompt assets used by the pipeline
- baseline reproduction material for the HCOME / X-HCOME / SimX-HCOME /
  Sim-HCOME family of workflows
- evaluation scripts and experimental setup artifacts
- example outputs and token/accounting summaries from prior experiments

The repository has been sanitized for open-source release. Real API keys,
private endpoints, machine-specific IDE state, and local absolute paths are
not included.

## What This Repository Contains

- `Code/new_code/`
  Main DHTW implementation.
- `Code/prompt/`
  Prompt templates used during KG exploration and KG construction.
- `Code/CodewithCollaborative copilot/`
  Variant of the pipeline with collaborative copilot integration.
- `Code/Evaluation code/`
  Evaluation and comparison scripts.
- `Data/EXPERIMENTAL SETUP/`
  Experimental inputs, baseline setup, and reproducibility materials.
- `Data/EXPERIMENTAL RESULTS/`
  Generated results, token summaries, schema artifacts, and prior run outputs.

## Repository Layout

```text
.
|-- Code/
|   |-- config.yaml
|   |-- new_code/
|   |-- prompt/
|   |-- seed_data/
|   |-- all_data/
|   |-- output/
|   |-- Evaluation code/
|   `-- CodewithCollaborative copilot/
|-- Data/
|   |-- EXPERIMENTAL SETUP/
|   |   `-- Baseline/hcome_reproduction/
|   `-- EXPERIMENTAL RESULTS/
|-- requirements.txt
`-- README.md
```

## Environment Requirements

- Python 3.12 is recommended
- Windows paths are used throughout the existing scripts and outputs
- The project depends on the packages listed in `requirements.txt`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Configuration

The main pipeline reads its runtime configuration from:

- `Code/config.yaml`
- `Code/CodewithCollaborative copilot/config.yaml`

The repository is now configured for safe open-source usage:

- `OpenAI_API_Base` is blank by default
- `API_key_list` uses environment-variable references
- no literal API key is stored in tracked config files

## Key Features

- Human-LLM collaborative KG exploration and construction workflow
- Research-oriented prompts and configuration for domain-specific extraction
- Baseline reproduction material for HCOME-family methods
- Evaluation scripts for comparing extracted triplets and graph outputs
- Preserved experimental artifacts to support reproducibility and inspection

### Recommended Environment Variables

Set at least:

```bash
OPENAI_API_KEY=your_api_key
```

Optional:

```bash
OPENAI_API_BASE=https://api.openai.com/v1
```

The runtime loader will:

- resolve `API_key_list` entries as environment variable names when possible
- fall back to `OPENAI_API_KEY` if referenced in config
- use `https://api.openai.com/v1` when no explicit base URL is provided

## Main DHTW Pipeline

The primary implementation lives in `Code/new_code/`.

Main entry point:

```bash
python Code/new_code/main.py
```

Important notes:

- `Code/new_code/main.py` loads `Code/config.yaml`
- the current entry script is structured to support both KG exploration and
  KG construction
- depending on your experiment, you may enable or disable stages inside
  `main.py`
- outputs are written under `Code/output/`

Core modules:

- `kg_exploration.py`
  Extracts entities, relations, labels, schema candidates, and statistics.
- `kg_construction.py`
  Builds KG triples and aligns constructed output with schema constraints.
- `main.py`
  Resolves runtime configuration, orchestrates pipeline execution, and writes
  aggregate statistics.

## Collaborative Copilot Variant

An additional implementation is available in:

```text
Code/CodewithCollaborative copilot/
```

Entry point:

```bash
python "Code/CodewithCollaborative copilot/new_code/main.py"
```

This variant includes collaborative copilot-specific prompt hooks and output
paths while preserving the same environment-variable-based secret handling.

## Baseline Reproduction

The baseline reproduction package is located at:

```text
Data/EXPERIMENTAL SETUP/Baseline/hcome_reproduction/
```

It provides four runnable levels:

- `level0_hcome.py`
- `level1_xhcome.py`
- `level2_simxhcome.py`
- `level3_simhcome.py`

See also:

- `Data/EXPERIMENTAL SETUP/Baseline/HCOME_training_guide.txt`
- `Data/EXPERIMENTAL SETUP/Baseline/hcome_reproduction/README.md`

Typical baseline workflow:

```bash
cd "Data/EXPERIMENTAL SETUP/Baseline/hcome_reproduction"
python level0_hcome.py
python level1_xhcome.py
python level2_simxhcome.py
python level3_simhcome.py
```

Baseline configuration file:

- `Data/EXPERIMENTAL SETUP/Baseline/hcome_reproduction/config.yaml`

This file has also been sanitized and now uses `OPENAI_API_KEY`.

## Evaluation

The main evaluation script is:

```bash
python "Code/Evaluation code/Evaluation code.py"
```

This module includes:

- text alignment utilities
- triplet parsing and matching
- similarity-based comparison using BERT embeddings
- bootstrap and permutation-style significance analysis

If you use the evaluation script, check the expected input file paths inside
the script first, because the evaluation workflow is research-oriented rather
than packaged as a CLI.

## Data and Results

The repository keeps experimental inputs and previous outputs for
reproducibility.

### Experimental Setup

- `Data/EXPERIMENTAL SETUP/Data Preparation/`
  Source datasets and prepared text collections
- `Code/seed_data/`
  Seed examples used by the main pipeline
- `Code/all_data/`
  Full text collections used for construction

### Experimental Results

- `Data/EXPERIMENTAL RESULTS/RQ*_Result/`
  Result files organized by research question
- `Data/EXPERIMENTAL RESULTS/token/`
  Token accounting and run summaries
- `Data/EXPERIMENTAL RESULTS/Outputs of each method (for All_Texts)/`
  Outputs from multiple methods on the shared corpus

These artifacts are retained intentionally to support reproducibility. They
have been normalized for open-source publication, but they still reflect
historical experiments rather than a polished packaged benchmark.

## Prompts

Prompt templates used by the DHTW pipeline are stored under:

```text
Code/prompt/
```

Major prompt groups include:

- `kg_exploration/`
- `kg_construction/`

If you change prompt templates, keep the corresponding config paths aligned in
`Code/config.yaml` or the collaborative-copilot variant config.

## Open-Source Release Notes

This repository has been prepared for GitHub publication with the following
cleanup steps:

- removed literal API keys
- removed private API endpoints
- removed IDE workspace state and local machine paths
- switched runtime secret handling to environment variables
- translated repository-facing documentation and code comments to English
- normalized log and result artifacts that contained non-English text

## Reproducibility Notes

- Historical outputs are kept in the repository so readers can inspect prior
  experiments without rerunning every workflow immediately.
- Runtime configuration has been sanitized, so reproducing runs now requires
  setting your own API credentials through environment variables.
- Some outputs were produced before open-source cleanup, so filenames and
  folder structure reflect the original research workflow rather than a
  package-style release process.

## Suggested Quick Start

1. Install dependencies.
2. Set `OPENAI_API_KEY`.
3. Review `Code/config.yaml`.
4. Run `python Code/new_code/main.py`.
5. Inspect generated output under `Code/output/`.
6. Explore baseline workflows in
   `Data/EXPERIMENTAL SETUP/Baseline/hcome_reproduction/` if you want to
   reproduce the human-in-the-loop variants.

## Limitations

- The codebase is organized as a research repository, not a packaged Python
  library.
- Several scripts assume existing file layouts and may need path adjustment if
  you reorganize the repository.
- Some historical outputs are preserved for reproducibility and may not match
  the exact defaults of the current sanitized configuration.

## License

No license file is included in the current repository snapshot. If you plan to
publish this project on GitHub, add an explicit license before release.

## Citation

If you publish work based on this repository, cite the associated paper or
project description that introduced DHTW.
