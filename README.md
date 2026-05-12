# Advanced RAG Mastery Notes

These notes are compiled from the first 21 pages of the course's introductory handwritten slides.

## Agenda & Approach
* **Approach**: Identify the problems with Large Language Models (LLMs) -> Find ways to solve them.
* **Key Questions**: 
  * Why is RAG important?
  * What is RAG?
  * How is RAG implemented?

## Introduction to LLMs & GPT
* **The Turning Point**: November 2022 saw the launch of ChatGPT, which was intelligent, powerful, and fast. It impacted domains like Tech, Coding, Finance, and Healthcare.
* **Under the Hood**:
  * **GPT** stands on the **Transformer Architecture** (a Deep Learning Model), heavily utilizing the **Self-Attention Mechanism**.
  * **Text** is understood through **Language Patterns**.
  * **LLMs (Large Language Models)** are AI models that learn through massive **Training** phases to establish **Parameters (Weights and Biases)**.
  * They possess **NLU (Natural Language Understanding)** which gives them incredible generative capabilities.
  * They are trained on vast portions of the **Internet** (Articles, books, webpages, blogs, reviews, coding repos).
* **The Cost of Pretraining**: Pretraining takes weeks or even months and requires massive Compute, Manpower, Storage, Infrastructure, and Energy.

## How LLMs Work: Next Word Predictors
* **Pretraining Phase**: The model learns how language works (patterns and distributions) and builds its **Knowledge** into its **Parameters**.
* **Inference Phase**: 
  * It acts as a "Next word predictor".
  * **Query (Prompt) -> Model -> Response (Completion)**.
  * Example: `Prompt: "Hi how are you?" -> LLM -> Completion: "Can I help you"`

---

## The Four Major Problems in LLMs

### 1. Knowledge Cutoff
* **The Issue**: Models are frozen in time and cannot access recent information.
* **Why it happens**: Pretraining takes months. The dataset used for training an Artificial Neural Network (ANN) architecture is static. For example, if a model completes training in Dec 2024 using a dataset from Aug 2024, any events from 2025 are unknown to it. The model only has "Intrinsic" parameterised knowledge up to its cutoff.

### 2. Hallucinations
* **The Issue**: Models generate plausible but false information confidently.
* **Why it happens**: Because generative AI relies on Natural Language Understanding (NLU) to generate text, it doesn't always rely on verifiable facts. It will generate "False facts" that look "Real".

### 3. No Source Attribution
* **The Issue**: You cannot verify or trace where the model's information came from.
* **Why it happens**: During training, sources are dissolved into the "Knowledge Parameters". When you query the model, the response is generated from these parameters, not by citing a verifiable source.

### 4. No Private Data Access
* **The Issue**: Models cannot access company or proprietary documents.
* **Why it happens**: Models are pretrained almost exclusively on Public Data. Their parameters only reflect public knowledge.

---

## Solving the Problems: Adding Information to the Context

To solve these four problems without doing massive, computationally expensive retraining (which takes long training times on a static dataset), we can pass **Additional Knowledge**.

* **Knowledge Cutoff**: Parameters + *Additional Knowledge*
* **Hallucinations**: Parameters + *Additional Knowledge* forces the model to use correct knowledge instead of hallucinating.
* **No Source Attribution**: Parameters lack source attribution, but if we provide *Source Information* explicitly, we can track it.
* **No Private Data Access**: Provide the private data as the *Additional Knowledge*.

### In-Context Learning (ICL) & The Context Window
* **Total Knowledge** = Parametric Knowledge (Internal) + External Knowledge.
* **Context**: What the model currently knows.
* **In-Context Learning (ICL)**: We can add additional/external knowledge to our model directly through the prompt. 
  * **Query + Additional Context = Better Answer**.
  * Example: Instead of just asking "What is my name?", you pass `Context: "My name is Himanshu."` + `Query: "What is my name?"`. The model uses the context to answer correctly.
* **Context Window Limits**: Models have context limits (e.g., 200k, 1M tokens). If you put too much information (like a 1000-page document) into the Transformer's self-attention mechanism, you suffer from the **"Lost in the middle"** phenomenon where the model forgets or ignores data in the middle of the context.
* **Solution**: You need a **Filter** to extract only the relevant parts of the additional knowledge to fit efficiently into the context window.

## The Solution: RAG (Retrieval Augmented Generation)
RAG is the architecture that dynamically retrieves the filtered, relevant *Additional Knowledge* and augments the *Generation* process of the LLM.

---

## Other Techniques for Customization

Aside from RAG and In-Context Learning, how else do we adapt LLMs?
1. **Prompt Engineering**: Tweaking the input prompt.
2. **Fine-Tuning**: Modifying the actual Parametric Knowledge of the model.
   * Modifies weights for specific expertise (e.g., Healthcare, Legal).
   * Techniques include **PEFT, LoRA, and QLoRA**.
   * **Drawback**: It is very compute-heavy and requires significant ML expertise compared to ICL/RAG.

---

## Deep Dive: How RAG Works

### What is RAG? (Retrieval Augmented Generation)
RAG is a solution to the four intrinsic problems of LLMs. It works by combining **In-Context Learning (ICL)** with a **Filtered Context Window**.
Instead of cramming an entire knowledge base into the context window (which leads to the "Lost in the Middle" problem), RAG dynamically filters and retrieves only the most relevant documents.

**The Process (Content Assembly)**:
1. **Knowledge Base**: Contains your updated, external knowledge (e.g., your private data).
2. **Query**: The user asks a question.
3. **Retrieval**: A "Smart Retriever" dynamically filters the knowledge base by checking the semantic meaning of the query against the documents. It finds the "Most similar" documents.
4. **Augmentation**: The retrieved documents are combined with the input prompt (`Input Prompt + External Retrieved Knowledge`).
5. **Generation**: The augmented prompt is fed to the Generative AI (LLM) to produce a highly relevant answer.

### Semantic Similarity
* To retrieve the right documents, the system uses **Semantic Similarity**.
* It calculates a **Similarity Score** between the query and each document.
* Documents are sorted in descending order by score, and the top most similar documents (e.g., Top 4) are selected and the rest are filtered out.

---

## The RAG Pipeline

How do we build this system for our Private Data? The pipeline consists of 4 main steps:

### 1. Document Loading
* **Knowledge Source**: The input of your external knowledge. This could be external files like PDF, HTML, CSV, DOC, TXT, or data from Google Drive, AWS S3 buckets, etc.
* **Document Loaders**: These loaders take the raw files, load them into memory, and parse them.
* **Parsing**: Extracts the structure of the document (Text, Tables, Images) and **Metadata** (Type, Name, Date of Creation, Size).

### 2. Chunking
* Passing a 100-page PDF directly is inefficient.
* **Chunking** breaks a large document into smaller, manageable parts (e.g., page-wise chunking).
* 1 PDF of 100 pages becomes 100 separate document chunks, each with its own metadata `{page no, document name}`.

### 3. Embedding
* **Text -> Numbers**: Computers don't understand text natively; they understand numbers.
* Deep Learning / NLP models (like BoW, TF-IDF, or advanced Embedder Models) convert text chunks into vectors.
* **Fixed Output Dimensions**: Regardless of whether a chunk has 500 or 5000 words, the embedder model converts it into a vector of a fixed dimension (e.g., 128 dimensions).
* *Note: Document Loading, Chunking, and Embedding are the most time-consuming steps in preparing the knowledge base.*

### 4. Vector Database
* The 100 document chunks are converted into 100 vectors.
* These are stored in a **Vector Store / Database** (Knowledge Base) which supports CRUD operations.
* **Storage Format**: `Vectors + Text + Metadata`.

---

## Putting It All Together: The Retrieval Phase

1. **User Query**: The user asks a question (e.g., "What is the company's policy on X?").
2. **Query Vector**: The query is passed through the *same Embedder Model* to generate a 128-dimensional query vector.
3. **Similarity Search**: The database plots the Query Vector and the Stored Vectors in a high-dimensional space. Vectors with similar semantic meanings will group together (have similar numbers).
4. **Ranking**: The database ranks the chunks by similarity (e.g., Chunk 1 = 0.9, Chunk 3 = 0.7, Chunk 2 = 0.2).
5. **Augmentation**: The top relevant chunks (Text + Metadata) are retrieved and assembled into a **Final Prompt** alongside the original query.
6. **Generation**: The final prompt goes to the LLM, producing the final response. *Guardrails* can be applied to the output to ensure safety.

---

## Recap: RAG Solves the 4 Problems

1. **Knowledge Cutoff**: Solved. We can dynamically add the latest events to the Knowledge Base, which are then retrieved.
2. **Hallucinations**: Solved. The LLM is grounded in facts provided in the retrieved context.
3. **No Source Attribution**: Solved. Because we store and retrieve Metadata alongside the text, we can point exactly to where the answer came from.
4. **No Private Data Access**: Solved. We load our private data into the Vector Store.

### Failures in RAG Systems
If RAG fails, it usually happens in one of two places:
1. **Retriever Failure**: The retriever fetches "garbage" or improper context from the knowledge base. If the context is wrong, the answer will be wrong.
2. **Generator Failure**: The context was retrieved correctly, but the LLM failed to understand or generate a coherent answer from the provided In-Context Learning (ICL).
