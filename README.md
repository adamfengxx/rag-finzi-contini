# Il Giardino dei Finzi-Contini RAG

A retrieval-augmented generation project for question answering on Giorgio Bassani's *Il giardino dei Finzi-Contini*.

## Overview

This project turns the novel into a searchable knowledge base:

- EPUB -> Markdown conversion *(Caution on copyright)*
- text cleaning and chapter-aware chunking
- embedding and indexing in ChromaDB
- dense retrieval + reranking
- grounded answer generation
- RAGAS-based evaluation

## Tech Stack

- Python
- LangChain
- ChromaDB
- Sentence Transformers / Hugging Face
- OpenAI API
- RAGAS
- BeautifulSoup, EbookLib, html2text

## Models 

**To adapt to the italian language contents**

- Embeddings: `intfloat/multilingual-e5-large`
- Reranker: `BAAI/bge-reranker-v2-m3`
- Generator: `gpt-4.1-mini`

## Retrieval Pipeline

The system uses a two-stage conversational RAG pipeline:

1. Query rewriting: recent user questions are used to rewrite follow-up questions into a self-contained retrieval query.
2. Dense retrieval: the rewritten query is searched in ChromaDB using E5 embeddings. **CosineSimilarity**
3. Reranking: the top 10 retrieved chunks are rescored with a cross-encoder reranker, and the best 4 are passed to the LLM.

Important design choice:
- conversation history is used only to resolve references **Ex: Who is she in the context**
- old assistant answers are never reused as evidence **To avoid the noise and impact on the accuracy of the anwer**
- final answers are grounded only in the freshly retrieved chunks

## Chunking

The project uses chapter-aware chunking instead of a simple fixed sliding window.

- `chunk_size = 700`
- `chunk_overlap = 120`

Each chunk keeps metadata such as **chapter, section, chunk id, and character offsets.**

## Evaluation 

The evaluation dataset is composed of two parts, 10 AI generated questions and 6 questions madde by Italian native speaker (of course this novel is his favorite book).
But from the evaluation results, the answers (the ground) given by Italian native speaker are too short while LLm tends to give relatively long answers, which makes the accucy high but 
relevance low. So, this part is still undergoing ...
