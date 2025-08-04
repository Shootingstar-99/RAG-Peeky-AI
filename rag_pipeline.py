from dotenv import load_dotenv
import os
from langchain.vectorstores.base import VectorStoreRetriever
from typing import List, Tuple
from pydantic import Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from prompt import CUSTOM_PROMPT

LLM_MODEL = "meta-llama/llama-3.3-70b-instruct:free" # Default LLM used with OpenRouter

load_dotenv() # Load environment variables from .env

class ThresholdRetriever(VectorStoreRetriever):
    """
    Custom retriever class that retrieves documents from the vector database
    only if their similarity score meets a minimum threshold.
    """
    threshold: float = Field(default=0.5)

    def _get_relevant_documents(self, query: str):
        """
        Perform vector similarity search with scores, keep documents above threshold.
        """
        docs_scores: List[Tuple] = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=7 # Retrieve top 7 results with their scores
        )
        return [document for document, score in docs_scores if score >= self.threshold]


class BuildRagChain:
    """
    Main RAG pipeline builder. Handles vectorstore loading, retriever configuration,
    and LLM+retrieval fusion via RetrievalQA for Q&A workflows.
    """

    def __init__(self, collection_name: str, database_path: str, embedding_model_name: str, llm_model= LLM_MODEL):
        self.collection_name = collection_name
        self.database_path = database_path
        self.embedding_model_name = embedding_model_name
        self.llm_model = llm_model

    def load_vectorstore(self):
        """
        Load a Chroma vectorstore using the designated embedding model and path.
        """
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            #model_kwargs={"device": "cuda"}, Use GPU for speed [I used Nvidia RTX 4060 cuda 12.6]
            encode_kwargs={"normalize_embeddings": True},
        )

        return Chroma(
            collection_name= self.collection_name,
            embedding_function=embeddings,
            persist_directory= self.database_path,
        )

    def build_rag_chain(self):
        """
        Assembles the entire RAG workflow:
         - Loads vectors
         - Sets up retriever with thresholding
         - Sets up the chat LLM (via OpenRouter)
         - Connects everything with a custom prompt

        Returns:
            RetrievalQA: LangChain retrieval-augmented generation chain.
        """
        print("Setting up LLM with RAG pipeline...")
        vectorstore = self.load_vectorstore()

        # Use custom retriever to filter weak matches
        retriever = ThresholdRetriever(vectorstore=vectorstore, threshold=0.42)
        # Set up LLM, using model and credentials from .env
        llm = ChatOpenAI(
            model= self.llm_model,
            temperature=0.7, # Controls response randomness
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",                # "stuff" = simple context
            chain_type_kwargs={"prompt": CUSTOM_PROMPT},
            return_source_documents=True, # Also returns docs used for answer provenance
        )
        return rag_chain
