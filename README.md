# RKG-VLM
### Retrieval-Enhanced Knowledge Graph Reasoning for Vision-Language Models
#### Improving VLM Physical World Reasoning via Knowledge Graphs

## Project Question
Alright, so this is the starting point for our 1-month project.<br>

><i>Can we improve VLM physical-world reasoning by giving it better, cleaner, and more relevant knowledge graph evidence?</i>

## What we're actually building

Firstly, we are **not reporducing the full VaLiK pipeline as baseline** instead:

- We use VaLiK as a **conceptual reference** for how a KG-augmented VLM pipeline can be structured
- We implement a **minimal version of that idea**, only including components relevant to our project

### High level Flow
```
Image + Question
        ↓
Entity Extraction
        ↓
Question Type Detection
        ↓
KG Node Mapping
        ↓
Candidate Retrieval (ConceptNet)
        ↓
Filtering (KG Quality)
        ↓
Reranking (Retrieval Quality)
        ↓
Top-K Evidence
        ↓
VLM (Qwen2.5-VL)
        ↓
Answer + Trace
```

## Core focus

### ***1. Retrieval Quality***
This is where we would spend the most efforts, to acheive this we must 
- make the retrieval **Question-Aware**
- select **relevant triples**
- avoid noise in data
- keep the **evidence and citations small and usefull** 
### ***2. KG Evidence Quality***
Ideally wo aren't building a new KG as our pipeline is already inspired from VaLiK, we only focus on:
- which relations we allow
- removing weak/generic edges
- keeping physically meaningful triples
### ***3. Interpretation***
So to be able to showcase some efforts in interpretability of the improved VLM we can provide a **support trace** like *entities*, *mapped nodes*, *retrived triples* and *final evidence*

## Rough Plan
For the project plan, I thought of keeping it initially defined into modular phases, we can update it to be more concurrent if need be to work on multiple parts simultaneously
<details>
<summary>Click to view the plan</summary>

## Phase 1 - Setup

- [x] Basic repo scaffold
- [x] runnable demo pipeline
- [x] initial cli & config
- [ ] Dataset selection 

## Phase 2 - baseline & Naive KG

- [ ] Integrate ***Qwen2.5-VL***
- [ ] Simple Entity Extraction (NER)
- [ ] Basic ConceptNet retrieval

## Phase 3 - Retrieval improvement

- [ ] Question classification
- [ ] Relation-aware filtering
- [ ] reranking
- [ ] top-k selection tuning 

## Phase 4 - KG Quality Filtering

- [ ] Remove weak edges
- [ ] restrict relations
- [ ] cleaner subgraphs 

## Phase 5 - Evaluation & Analysis

- [ ] Compare VLM only, Naive KG, Improved Retrieval
- [ ] Error Analysis
- [ ] Ablation

##### Further ahead we can go for Interpretability picking up the trace evidecnes from the graphs and using a small local LLM for cleaner demo of explanation.

</details>

## Stacks

- **Storage:** JSON initially, SQLite or Neo4j if we need graph traversal
- **Project management:** Python module => ***uv***
- **pipeline management:** Huggingface Transformers
- **VLM baseline model:** Qwen2.5-VL
- **Retrieval:** ConceptNet API
- **Storage:** Initially JSON, then we'll figure it out

## Usage
`uv sync` for the automatic venv setup\
`uv run <commands> <subcommand>` for running the file
