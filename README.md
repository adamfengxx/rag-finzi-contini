# Il Giardino dei Finzi-Contini RAG

A retrieval-augmented generation project for question answering on Giorgio Bassani's *Il giardino dei Finzi-Contini*.

## Overview

This project turns the novel into a searchable knowledge base:

- EPUB -> Markdown conversion
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

**To be adapted to the italian language contents**

- Embeddings: `intfloat/multilingual-e5-large`
- Reranker: `BAAI/bge-reranker-v2-m3`
- Generator: `gpt-4.1-mini`

## Retrieval Pipeline

The system uses a two-stage conversational RAG pipeline:

1. Query rewriting: recent user questions are used to rewrite follow-up questions into a self-contained retrieval query.
2. Dense retrieval: the rewritten query is searched in ChromaDB using E5 embeddings.
3. Reranking: the top 10 retrieved chunks are rescored with a cross-encoder reranker, and the best 4 are passed to the LLM.

Important design choice:
- conversation history is used only to resolve references
- old assistant answers are never reused as evidence
- final answers are grounded only in the freshly retrieved chunks

## Chunking

The project uses chapter-aware chunking instead of a simple fixed sliding window.

- `chunk_size = 700`
- `chunk_overlap = 120`

Each chunk keeps metadata such as chapter, section, chunk id, and character offsets.
