# Evaluation of Causal Reasoning for LLMs in Clinical Lab Test Scenarios

> Code & reproducible materials for the manuscript: **“Evaluation of Causal Reasoning for Large Language Models in Contextualized Clinical Scenarios of Laboratory Test Interpretation.”**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license) [![Reproducible](https://img.shields.io/badge/Reproducible-Yes-blue.svg)](#reproducibility--determinism) [![R 4.3](https://img.shields.io/badge/R-4.3+-blue)](#r-dependencies) [![Python 3.10](https://img.shields.io/badge/Python-3.10+-yellow)](#python-dependencies)

---

## tl;dr

We benchmark **GPT-o1** and **Llama-3.2-8B-Instruct** on **99** lab-test scenarios mapped to **Pearl’s Ladder** (Association, Intervention, Counterfactual) across **8** common tests (HbA1c, Creatinine, Vitamin D, CRP, Cortisol, LDL, HDL, Albumin). Four clinicians rated **binary correctness** and **reasoning quality**. We report AUROC/precision/recall/specificity plus **Fleiss’ kappa (Conger)** and **Kendall’s W**. This repo contains data schemas, prompts, scripts, and figure code to fully reproduce tables/plots.

---

## Repository Structure

```
.
├─ config/
│  ├─ config.yaml
│  └─ prompts/
│     ├─ causality_fewshot.txt
│     └─ examples.tsv
├─ data/
│  ├─ DataSet.xlsx
├─ scripts/
│  ├─ Causality.py
│  ├─ IRRFile.ipynb
│  ├─ Accuracy R Code.Rmd
│  ├─ IRR_Causality.Rmd
│  └─ IRR_Causality_4Raters.Rmd
│  └─ utils/
├─ results/
│  ├─ metrics/
│  ├─ figures/
│  └─ tables/
├─ env/
│  ├─ environment.yml
│  └─ r-packages.R
├─ notebooks/
│  ├─ Causality.ipynb
│  ├─ Accuracy R Code.nb.html
│  ├─ IRR_Causality.nb.html
│  └─ IRR_Causality_4Raters.html
├─ requirement.txt
└─ README.md
                 
```

---

## Installation

### 1) Clone

```bash
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>
```

### 2) Python environment (conda)

```bash
conda env create -f env/environment.yml
conda activate lab-causality
```

**Python dependencies** (subset): `pandas numpy matplotlib seaborn pyyaml tqdm openpyxl requests`

### 3) R dependencies

```r
source("env/r-packages.R")
```

---

## Configuration

All run-time settings live in `config/config.yaml`.

```yaml
paths:
  data_raw: "data/raw"
  data_processed: "data/processed"
  results: "results"
  figures: "results/figures"
  tables: "results/tables"
  prompts: "config/prompts"

random:
  seed: 42

questions:
  n_total: 99
  rungs: ["association", "intervention", "counterfactual"]
  lab_tests: ["HbA1c", "Creatinine", "Vitamin D", "CRP", "Cortisol", "LDL", "HDL", "Albumin"]

models:
  - name: "gpt-o1"
    provider: "openai"
    temperature: 0.0
    top_p: 1.0
    api_key_env: "OPENAI_API_KEY"
  - name: "llama-3.2-8b-instruct"
    provider: "local"
    endpoint: "http://localhost:1337/v1"
    temperature: 0.0
    top_p: 1.0
```

---

## End-to-End Reproduction

```bash
python scripts/01_generate_questions.py --config config/config.yaml
python scripts/02_run_models.py --config config/config.yaml --model gpt-o1
python scripts/02_run_models.py --config config/config.yaml --model llama-3.2-8b-instruct
python scripts/03_postprocess_outputs.py --config config/config.yaml
python scripts/04_collate_ratings.py --config config/config.yaml
Rscript scripts/05_compute_metrics.R
python scripts/06_plot_heatmaps.py
Rscript scripts/07_export_tables.R
```

---


---

## Reproducible Code

This repository includes all R and Python scripts used in the manuscript's analysis.

### **R Markdown Files**
- **Accuracy R Code.Rmd** – Computes binary accuracy metrics (AUROC, Precision, Sensitivity, Specificity) for each model and causal rung.
- **IRR_Causality.Rmd** – Calculates inter-rater reliability (IRR) statistics for binary and Likert-scale ratings (Fleiss’ kappa with Conger’s exact method, Kendall’s W).
- **IRR_Causality_4Raters.Rmd** – Extended IRR calculations using all four raters, including subgroup analyses.

### **Python Notebooks**
- **Causality.ipynb** – Generates causal reasoning questions, runs LLM queries, processes outputs, and formats for evaluation.
- **IRRFile.ipynb** – Python-based processing of rater files, merging outputs, and preparing input for R IRR analyses.

These scripts and notebooks allow **full reproduction** of:
1. Question generation
2. Model querying (GPT-o1 & Llama-3.2-8B-Instruct)
3. Post-processing outputs
4. Integrating clinician ratings
5. Computing accuracy metrics & inter-rater agreement
6. Generating all figures and tables in the manuscript

To reproduce results:
- Run `Causality.ipynb` → `IRRFile.ipynb` in Python
- Then run `Accuracy R Code.Rmd` → `IRR_Causality.Rmd` or `IRR_Causality_4Raters.Rmd` in R

All paths and settings can be updated in the provided configuration files.

---

## License

MIT License
# LLM_Causality_LabTest
Evaluation of Causal Reasoning for Large Language Models in Contextualized Clinical Scenarios of Laboratory Test Interpretation
