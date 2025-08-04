# 🤖 Peeky - Intelligent Document Assistant

Peeky is an advanced, fully object-oriented, modular, open-source RAG (Retrieval-Augmented Generation) pipeline that allows you to upload documents (PDFs), automatically clean, chunk, vectorize, and semantically search them with state-of-the-art LLMs via a modern Streamlit frontend. Peeky is designed for speed, scalability (leveraging NVIDIA RTX 4060 GPU for embeddings), and a seamless, ChatGPT-style UX—perfect for both technical and non-technical users.

## 🚀 Features
- **End-to-End RAG pipeline:**  
Fetch → Clean → Chunk (parallel, spaCy based) → Embed (GPU) → Store in ChromaDB → Natural Language QA with your chosen LLM.
- **Streamlit Frontend:**  
  - Modern, sleek UI with seamless chat (typewriter effect, source display).
  - Instant file upload, processing status, and document stats.
  - LLM selector present.
  - A warm coffee based theme.
- **Object-Oriented Python Design:**  
  - ```Fetch (fetch.py)```: Handles PDF discovery/page extraction.
  - ```DocumentCleaner (parse_and_clean.py)```: Cleans, normalizes, de-noises, and validates all content.
  - ```ChunkSuperFast (chunker.py)```: Multi-core, high-speed chunking that preserves sentence boundaries.
  - ```EmbedChunksAndStoreInChromaDB (embed_and_store_in_db.py)```: Fast, GPU-enabled batch embedding and vector storage.
  - ```BuildRagChain (rag_pipeline.py)```: Assembles custom retriever, embedding, vectorstore and LLM into a chain.
- **Multi-Model Flexibility**:  
  - Easily configure and swap LLM models via the sidebar.
  - Supports (Meta Llama 3.3, Deepseek R1, Google Gemma, Mistral 7B)
- **Conversational AI Enhancements:**:
  - Handles both context-based Q&A and basic social/utility dialogue.
  - Typewriter-reveal, streaming answers, markdown formatting.

## 💻 Computer & Performance
**Specs used while building the project:**
- **GPU:** NVIDIA GeForce RTX with CUDA 12.6
- **RAM:** 16GB
- **OS:** Windows (with multiprocessing and CUDA support verified)  
The pipeline exploits both multi-core CPU parallelism(for fast chunking/cleaning) and fast GPU acceleration (for SentenceTransformer embeddings).

## 📚 Directory Structure (OO Components)
```text
peeky/
├── app.py
├── fetch.py
├── parse_and_clean.py
├── chunker.py
├── embed_and_store_in_db.py
├── rag_pipeline.py
├── prompt.py
├── pdf_files/
├── peeky_database/
├── dbnameiteration.txt
├── .streamlit/
└── README.md
```

## Installation & Setup
1. **Requirements**  
   - Python 3.10 or newer
   - spaCy + en_core_web_sm
   - Streamlit
   - tqdm
   - chromadb
   - langchain, langchain_huggingface, langchain_chroma, langchain_openai
   - sentence-transformers
   - CUDA Toolkit for GPU (Cuda enabled PyTorch)
2. **Environment Setup**
```bash
# Create your environment
python -m venv venv
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# (You may also need)
python -m spacy download en_core_web_sm
```
3. **Environment Variables**
Create ```.env``` file with your OpenRouter API key (for LLM access):
```text
OPENROUTER_API_KEY=your-openrouter-api-key-here
```
## 🏃‍♂️ Usage
**To run the application**
```bash
streamlit run main.py
```
- Upload PDFs via the sidebar.
- Click "Configure" to process, embed, and index all files for Q&A.
- Chat with your documents using natural language!
- Clear files or change models at any time.

## ⭐ Credits
- Built by **Snehit Raj**
- Uses **LangChain, spaCy, Sentence Transformers, ChormaDB, and Streamlit**.