from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document
import chromadb
from tqdm import tqdm

# Constants for embedding model and database naming
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5" #takes 512 tokens at once for embedding
DB_PATH = "./peeky_database" # Name of chromadb database path
BASE_COLLECTION_DATABASE_NAME = "PEEKY-DATABASE-" # Base name for the collection in chromadb
ITERATION_FILE_NAME = "dbnameiteration.txt"

class EmbedChunksAndStoreInChromaDB:
    """
    Handles embedding of document text chunks and storage of their embeddings
    (along with text) in a persistent ChromaDB vector database.
    """

    def __init__(self, chunks: List[Document], embedding_model= EMBEDDING_MODEL):
        # Initialize embedding model on GPU with normalization
        self.embedding_model = HuggingFaceEmbeddings(model_name = embedding_model,
                                                     #model_kwargs={'device': "cuda"},
                                                     encode_kwargs= {'normalize_embeddings': True})
        self.chunks = chunks # List of Document objects to process
        self.texts = [doc.page_content for doc in chunks] # Extract page contents for embedding
        self.embeddings = None # Placeholder for computed embeddings
        self.create_embeddings()
        self.database_name = ""
        self.get_database_name()
        self.batch_size = 5000 # ChromaDB has an upper limit of 5641 entries at one time, so we create batches
        self.db_path = DB_PATH
        self.chroma_client = chromadb.PersistentClient(path= DB_PATH)
        self.collection = self.chroma_client.get_or_create_collection(self.database_name)

    def create_embeddings(self):
        """
        Generate embeddings for all chunks using the initialized model.
        Stores result in self.embeddings.
        """
        print(f"Creating embeddings for {len(self.texts)} documents...")
        self.embeddings = self.embedding_model.embed_documents(self.texts)
        print("Embeddings created successfully")

    def get_database_name(self):
        """
        Reads the current collection iteration from file, increments it,
        sets self.database_name, and saves the updated iteration to file.

        Ensures a unique name for each ChromaDB collection used.
        This is done so that the memory from the previous upload of documents
        don't interfere with the current iteration of the chatbot.
        """
        base = BASE_COLLECTION_DATABASE_NAME
        with open(ITERATION_FILE_NAME, "r") as file:
            iteration = int(file.read())

        with open(ITERATION_FILE_NAME, "w") as file:
            file.write(str(iteration + 1))

        self.database_name = base + str(iteration)

    def save_in_chromadb(self):
        """
        Saves all embedded chunks into ChromaDB in batches.

        Each batch is assigned unique IDs and added to the persistent vector store
        under the initialized collection.
        """
        for i in tqdm(range(0, len(self.texts), self.batch_size), desc= "Saving in database: "):
            batch_texts = self.texts[i : i + self.batch_size]
            batch_embeddings = self.embeddings[i : i + self.batch_size]
            batch_chunks = self.chunks[i : i + self.batch_size]
            batch_ids = [f"doc_{j}" for j in range(i, i + len(batch_texts))]

            # Add text, embeddings, and IDs to the Chroma collection
            self.collection.add(
                documents=batch_texts,
                embeddings=batch_embeddings,
                ids=batch_ids
            )
