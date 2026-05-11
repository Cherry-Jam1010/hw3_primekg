# MHKG Analysis System

Mental Health Knowledge Graph Analysis System based on PrimeKG.

## Project Overview

This project focuses on building a mental-health-oriented knowledge graph analysis and question answering prototype from PrimeKG. It includes:

- Mental health subgraph extraction
- Graph statistics and visualization
- Knowledge graph embedding experiments with TransE and RotatE
- Disease classification experiments
- LLM hallucination detection prototype
- RAG-based QA prototype
- Flask web application for demonstration

## Repository Structure

```text
.
├─ mental_health_kg/   # Main project source code
├─ PrimeKG-main/       # Reference PrimeKG materials
├─ README.md           # Root repository summary
└─ docs/               # GitHub Pages site
```

## Main Features

### 1. Subgraph Extraction

Extracts mental-health-related entities and relations from PrimeKG, including diseases, symptoms, drugs, and phenotypes.

### 2. Graph Analysis

Provides:

- node type distribution
- relation type distribution
- degree distribution
- centrality analysis
- disease relation analysis

### 3. Embedding Models

Implements and evaluates:

- TransE
- RotatE

### 4. Downstream Tasks

- Disease category classification
- Hallucination detection experiment
- RAG-based QA workflow

### 5. Web Demo

Includes several Flask pages for:

- graph browsing
- QA demo
- hallucination testing
- experiment visualization

## Local Run

Enter the main project folder:

```bash
cd mental_health_kg
pip install -r requirements.txt
python main.py --web
```

Then open:

```text
http://localhost:5000
```

## More Details

See the full project summary in the repository root:

- [Root README](../README.md)
