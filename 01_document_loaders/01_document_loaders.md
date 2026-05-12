# 01 - Document Loaders

This module covers the first critical step in building a Retrieval-Augmented Generation (RAG) pipeline: **Document Loading**. Before we can chunk, embed, and store our data, we must first parse our raw external knowledge sources and load them into standard LangChain `Document` objects.

LangChain provides specialized loaders for different file types. Each loader takes raw data and transforms it into a standard format containing:
1. `page_content`: The extracted raw text.
2. `metadata`: Additional information about the source (e.g., file path, page number, row number, etc.).

## Project Structure

### `knowledge_source/`
This folder contains the raw data files we want our LLM to have access to. 
- `transformers.txt`: A plain text file containing Q&A about the Transformer architecture.
- `organizations.csv`: Tabular data containing organization details (Name, Website, Country, Industry, etc.).
- `apparels.json`: Structured JSON data representing an apparel product catalog.
- `attention_is_all_you_need.pdf`: A complex PDF research paper.

### `notebooks/`
This folder contains Jupyter Notebooks demonstrating how to use different LangChain loaders.

#### 1. `text_loader.ipynb`
- **Goal**: Load standard `.txt` files.
- **Loader**: `TextLoader` from `langchain_community.document_loaders`.
- **Note**: Ensure the file is valid UTF-8, or use `autodetect_encoding=True` when instantiating the loader to avoid `UnicodeDecodeError` exceptions when dealing with special characters (like smart quotes).

#### 2. `csv_loader.ipynb`
- **Goal**: Load tabular `.csv` files.
- **Loader**: `CSVLoader` from `langchain_community.document_loaders`.
- **Note**: Each row in the CSV is typically loaded as a separate Document, which makes it easy for the retriever to grab specific row data later on. 

#### 3. `json_loader.ipynb`
- **Goal**: Load structured `.json` files.
- **Loader**: `JSONLoader` from `langchain_community.document_loaders`.
- **Note**: Requires the `jq` dependency to parse the JSON schema and extract specific fields into the `page_content` and `metadata`.

#### 4. `pdf_loader.ipynb`
- **Goal**: Load `.pdf` documents.
- **Loader**: LangChain provides multiple options like `PyPDFLoader` (standard text extraction), `PDFMinerLoader`, or `PDFPlumberLoader` (better for tables/formatting). 
- **Note**: PDFs are split page-by-page. Each Document object typically represents one page, and the metadata will contain the specific page number.

#### 5. `web_loader.ipynb`
- **Goal**: Extract and load text directly from websites or URLs.
- **Loader**: `WebBaseLoader` (using `beautifulsoup4` under the hood).
- **Note**: This is highly useful for dynamically pulling documentation or articles directly from the internet into your knowledge base.

## Summary
The goal of this module is to take any raw, unstructured, or semi-structured data from the `knowledge_source` and convert it into a uniform array of `Document` objects. Once converted, the data is ready to be passed to the next stage of the RAG pipeline: **Chunking**.
