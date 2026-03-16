# Yaqeen RAG

This repository contains two Arabic Retrieval-Augmented Generation (RAG) systems:

## 1. Hadith RAG

Located in [hadith_rag](/c:/Users/d_tol/Desktop/Graduation%20Project/yaqeen-rag/hadith_rag).

- Focuses on authenticated hadith retrieval and grounded answer generation
- Uses hybrid retrieval with dense search, sparse search, and reranking
- Exposes an API and simple web UI for querying hadith content

## 2. Quran RAG

Located in [quran_rag](/c:/Users/d_tol/Desktop/Graduation%20Project/yaqeen-rag/quran_rag).

- Focuses on Quran ayah retrieval, tafsir-aware search, and citation-grounded answers
- Uses Chroma-based dense retrieval, BM25, reranking, and answer generation
- Exposes API endpoints for search, retrieval, and answering

## Structure

```text
yaqeen-rag/
├── hadith_rag/   # Hadith retrieval and generation system
├── quran_rag/    # Quran retrieval and generation system
└── Docs/         # Project notes, plans, and research documents
```

## Notes

- Each project has its own `README.md`, requirements, and implementation details
- Start with the README inside the specific RAG you want to run
