# 03 Embeddings Notes

This section explores different approaches to generating vector embeddings for text data, utilizing both local models (via Ollama) and cloud-based models (via OpenAI). These embeddings are a critical component for building robust Retrieval-Augmented Generation (RAG) systems.

## Common Concepts
Across both approaches, standard LangChain utilities are used to prepare the data:
- **Document Ingestion:** `PyPDFLoader` (`langchain_community.document_loaders`) is used to read documents.
- **Text Chunking:** `RecursiveCharacterTextSplitter` (`langchain_text_splitters`) is used to split large documents into smaller, meaningful segments (e.g., `chunk_size=300`, `chunk_overlap=50`).

---

## 1. Local Embeddings with Ollama (`ollaama_embedding.ipynb`)

Running embeddings locally ensures data privacy and incurs zero API costs.

- **Setup:** Requires the Ollama application. The model used is `embeddinggemma`, which must be pulled locally (`ollama pull embeddinggemma`).
- **Direct Ollama API:**
  - Generating embeddings for a single query or a list of documents: `ollama.embed(model="embeddinggemma", input=text)`.
  - **Custom Dimensions:** You can truncate the output embedding dimension by specifying the `dimensions` parameter (e.g., `dimensions=512`).
- **LangChain Integration:**
  - Uses the `OllamaEmbeddings` class from `langchain_ollama`.
  - Embed multiple document chunks using the standard `embed_documents(texts=...)` method.

---

## 2. Cloud Embeddings with OpenAI (`openai_embeddings.ipynb`)

Using OpenAI provides access to state-of-the-art embedding models with high performance.

- **Setup:** Requires an OpenAI API key loaded securely via environment variables (`dotenv`).
- **Models Used:** Demonstrates both `text-embedding-3-large` and `text-embedding-3-small`.
- **LangChain Integration:**
  - Uses the `OpenAIEmbeddings` class from `langchain_openai`.
  - Embedding a single query: `embed_query(text=query)`.
  - Embedding document chunks: `embed_documents(texts=text_documents)`.
- **Dimensionality Reduction:** Similar to the local approach, OpenAI's `text-embedding-3-large` model supports native dimensionality reduction. This is achieved by passing the parameter during instantiation: `OpenAIEmbeddings(model=MODEL_LARGE, dimensions=256)`.
