# DHTW: Domain-Specific Semantic-Rich Knowledge Graph Construction through Human-LLM Team Working

DHTW constructs domain schemas through iterative human–LLM collaboration
and uses the validated schemas for knowledge graph extraction.

## Repository Layout

```text
.
|-- Code/
|   |-- config.yaml
|   |-- new_code/
|   |-- prompt/
|   |-- seed_data/
|   |-- all_data/
|   |-- Evaluation code/
|   `-- CodewithCollaborative copilot/
|-- Data/
|   |-- EXPERIMENTAL SETUP/
|   |   `-- Data Preparation/
|   |       `-- hcome_reproduction/
|   `-- EXPERIMENTAL RESULTS/
|       |-- RQ1_Downstream_Extraction/
|       |-- RQ2_HITL_Schema_Comparison/
|       |-- RQ3_Ablation/
|       |-- RQ4_Transfer_and_Refinement/
|       |-- RQ5_Resources_and_Expertise/
|       |-- Repeated_Runs/
|       |-- Shared_Artifacts/
|       `-- Historical_Results/
|-- requirements.txt
`-- README.md
```

## Setup

Use Python 3.12 and install the dependencies:

```bash
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` and review `Code/config.yaml` for the model,
input files, and output paths. `OPENAI_API_BASE` can be set when using
a custom API endpoint.

## DHTW Pipeline

Run the main entry point:

```bash
python Code/new_code/main.py
```

The main modules are:

- `Code/new_code/kg_exploration.py`: schema exploration and human revision.
- `Code/new_code/kg_construction.py`: schema-guided KG extraction.
- `Code/new_code/main.py`: configuration and pipeline execution.

Prompt templates are in `Code/prompt/kg_exploration/` and
`Code/prompt/kg_construction/`. The collaborative-copilot variant is in
`Code/CodewithCollaborative copilot/`.

## Baselines and Evaluation

HCOME-family implementations are in
`Data/EXPERIMENTAL SETUP/Data Preparation/hcome_reproduction/`:

- `level0_hcome.py`: HCOME.
- `level1_xhcome.py`: X-HCOME.
- `level2_simxhcome.py`: SimX-HCOME.
- `level3_simhcome.py`: Sim-HCOME.

The participation guide is
`Data/EXPERIMENTAL SETUP/Data Preparation/HCOME_training_guide.txt`.

Evaluation code is in `Code/Evaluation code/Evaluation code.py`,
including triple matching, BERT similarity, permutation tests, and
bootstrap confidence intervals.

## Data and Experimental Results

Input data and seed texts are in
`Data/EXPERIMENTAL SETUP/Data Preparation/`.
Results in `Data/EXPERIMENTAL RESULTS/` follow the manuscript:

| Experiment | Directory | Contents |
| --- | --- | --- |
| RQ1: downstream KG extraction | `RQ1_Downstream_Extraction/` | API results across backbones; ChemProt, DrugProt, and FinReflectKG results (Tables I–IV). |
| RQ2: HITL schema comparison | `RQ2_HITL_Schema_Comparison/` | Schema outputs and interaction records, grouped by dataset and method (Table V). |
| RQ3: ablation | `RQ3_Ablation/` | Human revision, iterative guidance, and additional schema conditions (Table VI). |
| RQ4: transfer and refinement | `RQ4_Transfer_and_Refinement/` | Biomedical schema transfer and refinement-round results. |
| RQ5: resources and expertise | `RQ5_Resources_and_Expertise/` | Token and human-cost records, participant results, and review times (Tables VII–VIII and RQ5). |

Additional result groups:

- `Repeated_Runs/`: three DHTW GPT-4o API extraction runs (Section VI-D):
  `Run1/our_GPT4o.csv`, `Run2/Run2.csv`, and `Run3/Run3.csv`.
- `Shared_Artifacts/Full_Corpus_Outputs/`: full-corpus method outputs,
  grouped by dataset.
- `Historical_Results/City/`: city dataset results.

Within RQ4, `Schema_Transfer/` contains the biomedical result tables
with the `DHTW_drug_chem` and `DHTW_chem_drug` columns and transfer
token records; `Refinement_Rounds/` contains the round-specific outputs.

Within RQ5, `Token_and_Human_Cost/` contains schema-construction and
downstream resource records; `Participant_Expertise/` contains the
expert, trained, and informed groups' schemas, KGs, and `Time.csv`.

