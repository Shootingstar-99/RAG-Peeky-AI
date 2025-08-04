import os
from langchain_community.document_loaders import PyMuPDFLoader
from tqdm import tqdm

class Fetch:
    """
    Fetches pdf documents from a specified folder.
    Each pdf file is loaded page by page of type: List[Document]
    """

    def __init__(self, pdf_path: str):
        self.all_documents = [] # List to hold all fetched objects
        self.pdf_path = pdf_path # Path to the directory with the pdf files

    def fetch_documents(self):
        """
        Scans the specified folder for pdf files, loads every page as a Document,
        and accumulates all documents in a single list.
        Returns:
            list: List of Document objects, one per page across all pdf files
        """
        print("Looking for documents...")

        # Loop through all files in the directory
        for filename in tqdm(os.listdir(self.pdf_path), desc= "Fetching files"):
            if filename.endswith(".pdf"):
                file_path = os.path.join(self.pdf_path, filename)

                #Use PyMuPDFLoader to split the Document objects, one per page
                loader = PyMuPDFLoader(file_path)
                documents = loader.load() # Returns a list od Document objects (pages)
                self.all_documents.extend(documents)

        return self.all_documents
