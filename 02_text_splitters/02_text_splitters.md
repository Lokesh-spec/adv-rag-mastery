# 02 - Text Splitters

This module covers the second critical step in building a Retrieval-Augmented Generation (RAG) pipeline: **Chunking / Text Splitting**. 

After loading raw data into LangChain `Document` objects (covered in Module 01), the text is often too large to fit into the context window of an LLM or to be accurately represented by a single embedding vector. To solve this, we break large documents into smaller, meaningful chunks.

LangChain provides the `langchain-text-splitters` package, which contains several strategies for splitting text. A good text splitting strategy ensures that chunks are small enough to process, but large enough to retain their semantic meaning without losing important context (the "Lost in the Middle" problem).

## Project Structure

### `notebooks/`
This folder contains Jupyter Notebooks demonstrating how to implement various text splitting strategies.

#### 1. `01_charcter_text_splitter.ipynb`
- **Goal**: Split text based on a specific, fixed character separator.
- **Tool**: `CharacterTextSplitter`.
- **How it works**: It looks for a specific character (like a newline `\n` or a space) to divide the text into chunks of a predefined `chunk_size` (e.g., 1000 characters). You can also set a `chunk_overlap` so that adjacent chunks share some text, preserving context across the boundary.

#### 2. `02_recursive_text_splitter.ipynb`
- **Goal**: Split text recursively using a hierarchy of separators.
- **Tool**: `RecursiveCharacterTextSplitter`.
- **How it works**: This is generally the **recommended text splitter** for generic text. It tries to split on double newlines `\n\n` (paragraphs) first. If the resulting chunk is still too large, it falls back to single newlines `\n` (sentences), then spaces ` `, and finally individual characters `""`. This ensures paragraphs and sentences are kept intact whenever possible.

#### 3. `03_document_text_splitter.ipynb`
- **Goal**: Split text while respecting the specific structure of the document format (e.g., Markdown, HTML, Code, PDF structure).
- **Tools**: Format-specific splitters like `MarkdownHeaderTextSplitter`, `HTMLHeaderTextSplitter`, or code splitters (e.g., `Language.PYTHON`).
- **How it works**: Instead of blindly splitting on characters, these splitters understand the structure. For example, a Markdown splitter can split a document based on its headers (`#`, `##`) and append the header information into the chunk's metadata. This ensures semantic grouping (e.g., a whole section about "Installation" stays together).

## Core Concepts
- **Chunk Size**: The maximum size of a chunk. This can be measured in characters or **tokens** (using libraries like `tiktoken`).
- **Chunk Overlap**: The number of characters or tokens that overlap between consecutive chunks. Overlap helps prevent splitting sentences or related ideas awkwardly in half.
- **Metadata**: Just like in Document Loaders, preserving metadata during chunking is crucial. It allows the retriever to know exactly which document (and which page or section) a chunk came from.

## Summary
The goal of this module is to transform massive `Document` objects into bite-sized, contextually meaningful chunks. Once the text is split appropriately, the chunks are ready for the next stage of the pipeline: **Embeddings and Vector Stores**.
