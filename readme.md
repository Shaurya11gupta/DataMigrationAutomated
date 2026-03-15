# DataMigrationCrane

**Intelligent schema mapping and data migration** — discover joins, map source columns to target columns, and infer transformations using name similarity, value similarity, and an LLM-based candidate selector.

---

## Overview

DataMigrationCrane helps you migrate data between different schemas by:

1. **Discovering joins** — Automatically finds foreign-key relationships between source tables using value overlap, constraints, and semantic cues.
2. **Mapping columns** — Maps each target column to the best source column(s) and suggests the transformation type (rename, concat, fk_lookup, etc.).
3. **Visualizing pipelines** — Interactive web UIs for exploring the **Name Similarity Engine** and **Value Similarity Engine** that power the pipeline.

---

## Features

| Feature | Description |
|---------|-------------|
| **Join discovery** | JoinGraphBuilderV2 uses ValueSimilarity + ConstraintCompatibilityJoin to discover FK edges between tables. |
| **Candidate selection** | SchemaMatcherLLM (Phi-3-mini + LoRA) maps target columns to source columns and infers transform types. |
| **Name similarity** | 4-stage pipeline: Regex Split → Segmentation Model → Abbreviation Expansion → Embedding + Conflict Detection. |
| **Value similarity** | Column profiling (type, MCV, distinct ratio, entropy) and weighted scoring for categorical and numeric columns. |

---

## Quick Start

### 1. Main Application (Schema Mapping Pipeline)

Full end-to-end flow: source schema → join discovery → target schema → column mappings.

```bash
# Install dependencies (Flask, value_similarity_engine, etc.)
pip install flask numpy

# For LLM-based mapping, also install:
pip install -r requirements-llm-matcher.txt
```

```bash
python app.py
```

Open **http://127.0.0.1:5000**

- **Step 1:** Define source tables (JSON or visual builder)
- **Step 2:** View discovered join graph
- **Step 3:** Define target schema
- **Step 4:** See candidate mappings (source → target + transform type)

---

### 2. Name Similarity Engine

Interactive pipeline visualizer: compare two column names step-by-step (regex split, segmentation, abbreviation expansion, embeddings, conflict detection).

```bash
# Requires: sentence-transformers, torch (for segmentation model)
pip install sentence-transformers torch
```

```bash
python name_similarity_app.py
```

Open **http://localhost:5001**

- Enter two column names (e.g. `custId` vs `customer_id`)
- See the full pipeline: tokens, expansion sources, conflicts, final similarity
- Use the 12 quick-run examples to explore edge cases

---

### 3. Value Similarity Engine

Interactive column comparison: paste two columns of sample values and see how similarity is computed (type detection, MCV overlap, distinct ratio, etc.).

```bash
# Requires: numpy
pip install numpy
```

```bash
python value_similarity_app.py
```

Open **http://localhost:5002**

- Paste values for Column A and Column B (or use built-in examples)
- See type detection, histograms (numeric), MCV overlap (categorical), and final score
- Supports numeric and categorical columns

---

## Architecture

A flowchart of the full architecture (engines, join discovery, candidate selection) is available as an interactive HTML diagram:

**[Open Architecture Flowchart](architecture_flowchart.html)**

You can open `architecture_flowchart.html` in a browser to explore:

- User flow (4 steps)
- API endpoints
- Join Discovery Engine (JoinGraphBuilderV2, ValueSimilarity, ConstraintCompatibilityJoin)
- Candidate Selection Engine (SchemaMatcherLLM, Phi-3 + LoRA)
- Per-column mapping loop (prompt → LLM → parse → validate)

---

## Demo Videos

| Demo | Description | Watch |
|------|-------------|-------|
| **Main Application** | Full schema mapping flow: source → joins → target → mappings | [Video](https://veersalabs-my.sharepoint.com/:v:/g/personal/shaurya_gupta_veersatech_com/IQCTjOPEzLdkTbCj8des33HcAdPaSL-OnnGHKlueL62gx6M?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=4upBsc) |
| **Name Similarity Engine** | Pipeline visualization, abbreviation expansion, conflict detection | [Video](https://veersalabs-my.sharepoint.com/:v:/g/personal/shaurya_gupta_veersatech_com/IQAdUk0G7bHSSKVftomZehT9AVgBGvMbbX_g2fQqTLNOqSE?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=uqHm1i) |
| **Value Similarity Engine** | Column profiling, type detection, similarity scoring | [Video](https://veersalabs-my.sharepoint.com/:v:/g/personal/shaurya_gupta_veersatech_com/IQAf5FtFfTkNQL2OGfFKGp6uAXBchzJw5FIoH5EGRo9JAVU?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=obsD5w) |

---

## Port Summary

| App | Port | URL |
|-----|------|-----|
| Main application | 5000 | http://127.0.0.1:5000 |
| Name Similarity Engine | 5001 | http://localhost:5001 |
| Value Similarity Engine | 5002 | http://localhost:5002 |

Run each in a separate terminal. They can run simultaneously.

---

## Requirements

### Core (Main App, Name Engine, Value Engine)

- Python 3.8+
- Flask
- numpy
- `value_similarity_engine` (included)
- `seg_classf_abbrev_test` (for Name Similarity Engine)
- `sentence-transformers` (for Name Similarity Engine)
- `constraint_similarity_engine`, `join_graph_builder_v2`, `pipeline_bridge` (included)

### LLM Candidate Selection (Main App Step 4)

- `pip install -r requirements-llm-matcher.txt`
- Model: `schema_matcher_llm1` (Phi-3-mini + LoRA adapter) must be present in `schema_matcher_llm1/`
- Runs on CPU by default

---

## Project Structure

```
.
├── app.py                    # Main schema mapping web app (port 5000)
├── name_similarity_app.py    # Name Similarity Engine UI (port 5001)
├── value_similarity_app.py   # Value Similarity Engine UI (port 5002)
├── architecture_flowchart.html  # Architecture diagram
├── pipeline_bridge.py        # json_to_tables, discover_joins, run_full_mapping
├── join_graph_builder_v2.py  # Join discovery (ValueSimilarity, ConstraintCompatibilityJoin)
├── candidate_generation_v3.py # SchemaMatcherLLM (Phi-3 + LoRA)
├── value_similarity_engine.py # ColumnStats, ValueSimilarity
├── seg_classf_abbrev_test.py  # name_similarity pipeline
├── constraint_similarity_engine.py
├── schema_matcher_llm1/      # Fine-tuned LLM adapter
├── demo_videos/              # Demo walkthrough videos
├── templates/                # HTML templates
└── static/                   # CSS, JS
```

---

## Transformation Classifier (Optional)

For training a binary classifier on (source → target, transform_type) validity, see [readme.md](readme.md) in this repo. It covers:

- Data generation (`generate_full_transformation_data.py`)
- Unified dataset (`build_unified_training_data.py`)
- Training (`train_transformation_classifier.py`)

---

## License

[Add your license here]
