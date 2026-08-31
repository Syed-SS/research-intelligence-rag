# Research Intelligence RAG

A research-focused Retrieval-Augmented Generation (RAG) system designed to retrieve and generate answers from a collection of research papers, with systematic evaluation using RAGAS.

## Overview

Research Intelligence RAG combines retrieval, knowledge representation, and language-model generation to answer research-oriented questions using external research documents rather than relying only on the language model's internal knowledge.

The project organizes research papers into a structured RAG dataset and evaluates the resulting answers using four RAGAS metrics:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

The retrieval layer also uses LightRAG-based knowledge representation, including entity/relationship-oriented information and document-level evidence.

---

## Objectives

The main objectives of the project are to:

1. Build a research-oriented RAG pipeline using real research papers.
2. Organize the source material into reproducible evaluation datasets.
3. Generate research questions grounded in the collected papers.
4. Retrieve supporting evidence for each question.
5. Generate answers using the retrieved information.
6. Quantitatively evaluate the RAG system using RAGAS.
7. Investigate and resolve evaluation failures without altering the underlying research dataset.

---

## Project Dataset

The research collection is organized into:

```text
RAG-001
RAG-002
RAG-003
RAG-004
RAG-005
RAG-006
RAG-007
RAG-008
RAG-009
RAG-010

The final batch evaluation contains:

10 research-paper datasets
27 evaluation questions
4 RAGAS metrics
108 metric evaluations

The evaluation records contain:
user_input
retrieved_contexts
response
reference
faithfulness
answer_relevancy
context_precision
context_recall

System Architecture

The overall workflow is:
Research Papers
      |
      v
Document Processing
      |
      v
Knowledge Representation / LightRAG
      |
      v
Retrieval
      |
      v
Relevant Research Context
      |
      v
LLM Generation
      |
      v
Generated Answer
      |
      v
RAGAS Evaluation
      |
      +-------------------+
      |       |       |   |
      v       v       v   v
  Faith.    Answer   Context  Context
             Rel.   Precision Recall

LightRAG

LightRAG is used as part of the retrieval and knowledge-representation layer.

The retrieved information can contain structured knowledge-graph/entity information together with document-level evidence. This allows the system to use relationships and source information when constructing the context supplied to the generator.

RAG Evaluation Methodology

The final evaluation uses four RAGAS metrics.

1. Faithfulness

Measures whether the claims made in the generated response are supported by the retrieved context.

A higher score indicates that the generated answer is more grounded in the retrieved evidence.

2. Answer Relevancy

Measures how relevant the generated response is to the user's question.

A higher score indicates that the response better addresses the question being asked.

3. Context Precision

Measures the relevance/precision of the retrieved context with respect to the information required to answer the question.

4. Context Recall

Measures how much of the required information was successfully retrieved into the context.    
Final Evaluation Results

The final frozen evaluation contains 27 valid samples with zero NaN values across all four metrics.

Metric	Score

Faithfulness	0.7132
Answer Relevancy	0.9443
Context Precision	1.0000
Context Recall  

Evaluation completeness
Total samples          : 27
Total metrics          : 4
Total evaluations      : 108

Faithfulness NaN       : 0
Answer Relevancy NaN   : 0
Context Precision NaN  : 0
Context Recall NaN     : 0

Faithfulness Evaluation Issue and Resolution

During the batch evaluation, several Faithfulness evaluations returned NaN.

Investigation showed that the failure occurred during the structured-output/NLI stage of the RAGAS evaluation process, producing IncompleteOutputException for some responses.

The original RAG dataset and existing successful metric results were preserved.

A separate Faithfulness evaluation context was prepared, and the Faithfulness NLI statements were processed in smaller batches.

The repair process evaluated only the previously failed Faithfulness rows.

Original Faithfulness NaN rows : 17
Re-evaluated rows              : 17
Remaining NaN rows             : 0

The final repaired evaluation was written to a separate final JSON artifact.

Final Evaluation Artifact

The final verified evaluation file is:

ragas_batch_rag002_010_FINAL.json

This file contains the final results for all 27 samples and all four evaluation metrics.

The original evaluation result file was preserved separately during the repair process.
Technology Stack
Core
Python
Retrieval-Augmented Generation (RAG)
LightRAG
Large Language Models (LLMs)
Evaluation
RAGAS
RAGAS 0.4.3
Faithfulness
Answer Relevancy
Context Precision
Context Recall
Data / Processing
JSON
Research-paper documents
Structured retrieval contexts
Model Services
OpenAI API
GPT-4o-mini
text-embedding-3-small
Project Structure

A simplified project structure is:

research-intelligence-rag/
│
├── .venv/
│
├── RAG-001.txt
├── RAG-002.txt
├── ...
├── RAG-010.txt
│
├── ragas_batch_rag002_010_data.json
│
├── ragas_batch_rag002_010_results.json
├── faithfulness_evaluation_data.json
├── ragas_batch_rag002_010_faithfulness_fixed.json
├── ragas_batch_rag002_010_FINAL.json
│
├── run_ragas_batch_rag002_010.py
├── prepare_faithfulness_context.py
├── repair_faithfulness_final.py
│
└── README.md

Additional diagnostic/helper scripts may also be present in the development workspace.
Reproducibility

The evaluation process was designed to keep the original dataset separate from the repaired Faithfulness evaluation.

The final evaluation artifact can therefore be inspected independently without rerunning the full RAGAS evaluation.

The final JSON contains the individual evaluation results for each question as well as aggregate metric scores.

Limitations

The current evaluation has several limitations:

The evaluation set contains 27 questions.
The questions cover the selected research-paper collection rather than the full universe of RAG research.
RAGAS scores depend on the evaluator model and evaluation configuration.
Retrieval quality depends on the quality and structure of the underlying research documents.
A high aggregate metric does not guarantee that every individual response is perfect.
Future Improvements

Potential future improvements include:

Expanding the research-paper collection.
Increasing the number of evaluation questions.
Comparing LightRAG with alternative retrieval approaches.
Performing retrieval ablation studies.
Comparing different embedding models.
Evaluating different generator models.
Adding human evaluation alongside automated RAGAS evaluation.
Tracking evaluation results across different system configurations.
Conclusion

Research Intelligence RAG provides a structured workflow for research-oriented retrieval and generation, together with quantitative evaluation of the resulting RAG system.

The final evaluation covers 27 questions across four RAGAS metrics, with no missing metric values in the final evaluation artifact.

The final aggregate scores are:

Faithfulness       0.7132
Answer Relevancy   0.9443
Context Precision  1.0000
Context Recall     0.9630

The project therefore provides both a working research-oriented RAG workflow and a reproducible evaluation artifact for analyzing retrieval and generation quality.

