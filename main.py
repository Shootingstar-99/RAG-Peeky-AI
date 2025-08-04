import streamlit as st
import os
import time
from glob import glob
from fetch import Fetch
from parse_and_clean import DocumentCleaner
from chunker import ChunkSuperFast
from embed_and_store_in_db import EmbedChunksAndStoreInChromaDB
from rag_pipeline import BuildRagChain

# CONSTANTS AND PATH CONFIGURATION
PDF_FOLDER_PATH = "pdf_files"
DATABASE_PATH = "./peeky_database"
MAX_FILE_SIZE_MB = 200

os.makedirs(PDF_FOLDER_PATH, exist_ok=True)

# STREAMLIT PAGE CONFIGURATION
st.set_page_config(
    page_title="Peeky - Intelligent Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SESSION STATE INITIALIZATION

if "messages" not in st.session_state:
    #  Pre-loads an initial assistant greeting
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to Peeky! Please configure your settings and upload documents."}]

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None # For the RetrievalQA chain instance

if "system_ready" not in st.session_state:
    st.session_state.system_ready = False # True if backend is ready for Q&A

if "current_state" not in st.session_state:
    st.session_state.current_state = "Idle" # Current backend state

if "processing_log" not in st.session_state:
    st.session_state.processing_log = [] # Stores recent processing steps

if "selected_embedding_model" not in st.session_state:
    st.session_state.selected_embedding_model = "BAAI/bge-small-en-v1.5" # Default embedding model

if "selected_llm" not in st.session_state:
    st.session_state.selected_llm = "meta-llama/llama-3.3-70b-instruct:free" # Default LLM model

# UTILITY FUNCTIONS

def get_file_count():
    # Count number of PDF files in the upload directory.
    try:
        pdf_files = glob(os.path.join(PDF_FOLDER_PATH, "*.pdf"))
        return len(pdf_files)
    except:
        return 0

def update_status(status, log_message):
    # Update processing status and append to the log (Max last 10 entries).
    st.session_state.current_state = status
    if log_message:
        st.session_state.processing_log.append(f"{time.strftime('%H:%M')} - {log_message}")
        st.session_state.processing_log = st.session_state.processing_log[-10:]

# DOCUMENT PIPELINE: UPLOAD, CLEAN, CHUNK, EMBED, DB, and LLM CHAIN

def process_documents_pipeline(progress_placeholder, log_placeholder):
    """
    Full end-to-end pipeline:
     - Load PDFs
     - Clean their content
     - Chunk documents
     - Generate embeddings and store in ChromaDB
     - Prepare RAG chain with user-selected LLM and embedders
     - Fully updates status and logs at each stage
    """
    try:
        # First fetch pdfs
        update_status("Fetching Documents", "Scanning for pdfs...")
        progress_placeholder.progress(10)
        fetcher = Fetch(PDF_FOLDER_PATH)
        all_pdfs = fetcher.fetch_documents()
        if not all_pdfs:
            update_status("Error", "NO PDF documents found!")
            return False
        update_status("Fetching Documents", f"Found {len(all_pdfs)} documents")
        progress_placeholder.progress(20)

        # 2. Clean documents
        update_status("Cleaning Data", "Cleaning and processing...")
        progress_placeholder.progress(30)
        cleaner = DocumentCleaner()
        cleaned_documents = cleaner.clean_documents(all_pdfs)
        update_status("Cleaning data", f"Cleaned {len(cleaned_documents)} documents.")
        progress_placeholder.progress(40)

        # 3. Chunk cleaned documents
        update_status("Chunking Documents", "Creating chunks...")
        chunker = ChunkSuperFast(cleaned_documents)
        chunked_documents = chunker.chunk_pdfs_parallel(batch_size= 200)
        update_status("Chunking Documents", f"Created {len(chunked_documents)} chunks")
        progress_placeholder.progress(65)

        # 4. Embedding and storing in vector database
        update_status("Creating Embeddings", "Creating vector embeddings...")
        progress_placeholder.progress(75)
        embedder = EmbedChunksAndStoreInChromaDB(chunked_documents, embedding_model= st.session_state.selected_embedding_model)
        embedder.save_in_chromadb()
        update_status("Uploading to Database", "Storing embeddings in database...")
        progress_placeholder.progress(85)

        # 5. Create/load RetrievalQA chain (LLM + Retriever)
        update_status("Setting up LLM", "Initializing RAG system...")
        progress_placeholder.progress(95)
        try:
            with open("dbnameiteration.txt", "r") as file:
                latest_iteration = int(file.read()) - 1
            collection_name = f"PEEKY-DATABASE-{latest_iteration}"

            st.session_state.rag_chain = BuildRagChain(
                collection_name= collection_name,
                database_path= DATABASE_PATH,
                embedding_model_name= st.session_state.selected_embedding_model,
                llm_model= st.session_state.selected_llm
            ).build_rag_chain()
        except Exception as e:
            update_status("Error", f"Failed to initialize RAG chain: {str(e)}")
            return False

        # 6. Mark system as ready
        update_status("Ready", "✅ All set! You can start chatting now")
        progress_placeholder.progress(100)
        st.session_state.system_ready = True
        return True
    except Exception as e:
        update_status("Error", f"Pipeline failed: {str(e)}")
        return False

def process_existing_files():
    """
    Checks if there are PDF files present on start-up and automatically
    processes them if the system has not been initialized,
    ensuring the user can't chat until files are ready.
    :return:
    """
    file_count = get_file_count()
    if file_count > 0 and not st.session_state.system_ready:
        st.info(f"📄 Found {file_count} existing PDF files. Processing automatically...")
        progress_placeholder = st.progress(0)
        log_placeholder = st.empty()
        success = process_documents_pipeline(progress_placeholder, log_placeholder)
        if success:
            st.success("✅ Existing files processed successfully!")
        else:
            st.error("❌ Failed to process existing files. Check the logs.")
        progress_placeholder.empty()

#MAIN PAGE HEADING
st.title("🤖 Peeky", anchor= False)
st.subheader("Say yes to RAG-ing!", anchor= False)

# SIDEBAR: FILE UPLOAD, LLM CHOICE, STATS, CLEAR, CONFIGURE

# LLM selection widget (selects OpenRouter model string)
st.sidebar.subheader("Configuration Settings")
LLM = st.sidebar.selectbox("Choose your LLM: ",
                           ["Meta-Llama-3.3", "Deepseek-R1", "Google-Gemma-3-4B", "Mistral-7B"],
                           key= "llm_selector")
if LLM == "Google-Gemma-3-4B":
    st.session_state.selected_llm = "google/gemma-3-4b-it:free"
elif LLM == "Mistral-7B":
    st.session_state.selected_llm = "mistralai/mistral-7b-instruct:free"
elif LLM == "Deepseek-R1":
    st.session_state.selected_llm = "deepseek/deepseek-r1:free"
elif LLM == "Meta-Llama-3.3":
    st.session_state.selected_llm = "meta-llama/llama-3.3-70b-instruct:free"

# File uploader for pdf files, with size check
st.sidebar.subheader("📄 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB per file"
)

if uploaded_files:
    # Check file sizes and reject oversized ones
    total_size = sum([file.size for file in uploaded_files]) / (1024*1024)
    oversized = [file.name for file in uploaded_files if file.size > MAX_FILE_SIZE_MB * 1024 * 1024]

    if oversized:
        st.sidebar.error(f"The following files exceed the {MAX_FILE_SIZE_MB}MB limit: {', '.join(oversized)}")
    else:
        if st.sidebar.button("Upload Files", use_container_width= True):
            try:
                uploaded_count = 0
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(PDF_FOLDER_PATH, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    uploaded_count += 1

                st.sidebar.success(f"Uploaded {uploaded_count} files successfully!")
                st.session_state.system_ready = False
                time.sleep(1)
                st.rerun()

            except Exception as e:
                st.sidebar.error(f"Upload failed: {str(e)}")

# Sidebar file stats
st.sidebar.markdown("---")
st.sidebar.subheader("📊 File statistics")
file_count = get_file_count()
st.sidebar.markdown(f"### 📄 PDF Files: {file_count}", unsafe_allow_html= True)
st.sidebar.markdown("---")

# Clear all files feature, with confirmation dialog
st.sidebar.subheader("🗑️ Clear Files")
if file_count > 0:
    if st.sidebar.button("🗑️ Delete All Files", use_container_width=True, type="secondary"):
        try:
            pdf_files = glob(os.path.join(PDF_FOLDER_PATH, "*.pdf"))
            @st.dialog("Confirm deletion?")
            def confirm_delete():
                st.write(f"Are you sure you want to delete {len(pdf_files)} pdf files?")
                col1, col2 = st.columns(2)
                if col1.button("DELETE", type= "primary"):
                    delete_count = 0
                    for pdf_file in pdf_files:
                        os.remove(pdf_file)
                        delete_count += 1

                    st.session_state.system_ready= False
                    st.session_state.rag_chain = None
                    st.session_state.current_state = "Idle"
                    st.session_state.processing_log = []

                    st.success(f"Deleted {delete_count} files successfully!")
                    time.sleep(2)
                    st.rerun()

                if col2.button("Cancel!"):
                    st.rerun()
            confirm_delete()

        except Exception as e:
            st.sidebar.error(f"Delete failed: {str(e)}")

else:
    st.sidebar.info("📂 No files to delete")
st.sidebar.markdown("---")

# System status and processing logs
st.sidebar.subheader("💻 System Status")

if st.session_state.processing_log:
    with st.sidebar.expander("Processing Log", expanded= False):
        # Shows log entries in reverse chronological order (most recent first)
        for log_entry in st.session_state.processing_log[-1:-5:-1]:
            st.text(log_entry)

st.sidebar.markdown("---")

# Run the end-to-end pipeline on configure button
configure_button = st.sidebar.button(
    "⚙️ Configure",
    use_container_width= True,
    disabled= (file_count == 0),
    help = "Process all documents and set up the RAG system" if file_count > 0 else "Upload some PDF files first!"
)

if configure_button:
    st.session_state.system_ready = False
    progress_placeholder = st.progress(0)
    log_placeholder = st.empty()

    with st.spinner("Processing..."):
        success = process_documents_pipeline(progress_placeholder, log_placeholder)

    if success:
        st.success("✅ Configuration complete! PEEKY is ready.")
    else:
        st.error("❌ Configuration failed. Please check the logs.")

    progress_placeholder.empty()

# Startup: auto-process existing files if present and uninitialized
if file_count > 0 and not st.session_state.system_ready and st.session_state.current_state == "Idle":
    process_existing_files()
st.markdown("---")

# MAIN CHAT INTERFACE

# Chat history: display all messages so far (with sources)
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show sources (expandable) if present in assistant reply
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                with st.expander("🗃️ View sources", expanded= False):
                    for j, source in enumerate(message["sources"]):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source {j + 1}:</strong><br>
                            {source[:300]}...
                        </div>
                        """, unsafe_allow_html=True)

# Chat input: disables until system is fully ready
if not st.session_state.system_ready:
    st.chat_input("System not ready. Please configure first...", disabled= True)

    if file_count == 0:
        st.info("Please upload some PDF documents using the sidebar to get started.")
    elif st.session_state.current_state != "Ready":
        st.info("Please click 'Configure' in the sidebar to set up Peeky!")

# Main Q&A loop when system is ready
else:
    if query := st.chat_input("What would you like to know about your documents?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            try:
                # Run inference via chain; show spinner while waiting
                with st.spinner("Thinking..."):
                    result = st.session_state.rag_chain.invoke({"query": query})
                    assistant_response = result["result"]
                    sources = [doc.page_content[:400] + "..." for doc in result.get("source_documents", [])]

                # Typewriter effect for realism
                full_response = ""
                for char in assistant_response:
                    full_response += char
                    if char in [" ", '\n', '.', '!', '?']:
                        time.sleep(0.03)
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                # Store the assistant response and its sources
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources
                })

                # Display sources immediately beneath answer
                if sources:
                    with st.expander("🗃️ View Sources", expanded= False):
                        for j, source in enumerate(sources):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Source {j + 1}:</strong><br>
                                {source}
                            </div>
                            """, unsafe_allow_html=True)

            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
