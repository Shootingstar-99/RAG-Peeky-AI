import re
import html
from typing import List, Optional
from langchain.schema import Document
from tqdm import tqdm

"""
Keywords that indicate text should be skipped during cleaning.
This is done to make sure that certain terms and clauses are skipped to make sure
only relevant data is kept.
"""
KEYWORDS_TO_SKIP = ("table of contents", "index", "references", "appendix", "page", "copyright",
                    "advertisement", "cookie", "privacy policy", "footer", "header", "subscribe",
                    "notification", "newsletter")
MIN_CONTENT_LENGTH = 50 # Minimum length for content to be considered relevant

class DocumentCleaner:
    """
    Cleans and preprocesses raw text documents to prepare for indexing or embedding.
    Steps include removing noisy lines, filtering irrelevant text by keywords and length,
    normalizing whitespaces, and HTML entity unescaping.
    """

    def __init__(self,
                 keywords_to_skip: Optional[tuple[str, ...]] = KEYWORDS_TO_SKIP,
                 min_content_length: int = MIN_CONTENT_LENGTH):
        self.keywords_to_skip = keywords_to_skip
        self.min_content_length = min_content_length

        # Default regex patterns to identify noise lines typically found in pdfs and scraped text
        default_patterns = [
            r"^\s*Page\s+\d+",  # Page numbers
            r"^\s*@\w+",  # Social media handles
            r"^\s*\d{4}[-/]\d{2}[-/]\d{2}",  # Dates
            r"^\s*Copyright\s+©",  # Copyright notices
            r"^\s*\d+\s*$",  # Standalone numbers
            r"^\s*[-=_]{3,}\s*$",  # Separator lines
        ]

        # Compile regex patterns for efficient repeated matching
        self.noise_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in
                               default_patterns]

    def is_relevant(self, text: str) -> bool:
        """
        Check if the given text is relevant based on length and absence of skip keywords.
        """
        if not text or len(text.strip()) < self.min_content_length:
            return False

        text_lower = text.lower()
        return not any(keyword in text_lower for keyword in self.keywords_to_skip)

    def remove_noise_lines(self, text: str) -> str:
        """
        Remove lines that match predefined noise patterns and empty lines.
        """
        if not text:
            return ""
        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:
            if not line.strip():
                continue # Skip empty lines

            # Check if the line matches any noise pattern
            is_noise = any(pattern.match(line) for pattern in self.noise_patterns)

            if not is_noise:
                cleaned_lines.append(line.strip())

        # Join all cleaned lines into a single line string separated by spaces
        return " ".join(cleaned_lines)

    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace characters in the text.
        """
        if not text:
            return ""
        # Replace various newline/tab characters with space
        text = text.replace("\\n", " ").replace("\\t", " ").replace("\n", " ")
        # Collapse multiple whitespace characters into a single space
        text = re.sub(r'\s+', ' ', text)

        return text.strip() # Trim leading and trailing spaces

    def clean_text(self, text: str) -> Optional[str]:
        """
        Fully clean a single text string by applying unescaping, noise removal, relevance filtering,
        and whitespace normalization.
        """
        if not text: return None

        try:
            # Unescapte HTML entities
            text = html.unescape(text)

            # Early relevance check - filter out very short or irrelevant text early
            if not self.is_relevant(text):
                return None

            # Remove noise lines like page numbers, social handles, separators and normalize whitespace to have clean continuous text
            text = self.normalize_whitespace(self.remove_noise_lines(text))

            # Final relevance check after cleaning
            if not self.is_relevant(text): return None

            return text

        except Exception as e:
            print(f"Error cleaning text: {e}")
            return None

    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """
        Clean a list of Document objects by cleaning their 'page_content'.
        """
        print("Cleaning documents...")
        if not documents:
            return []

        cleaned_docs = []

        for i, doc in tqdm(enumerate(documents), desc= "Cleaning documents: "):
            try:
                cleaned_text = self.clean_text(doc.page_content)

                if cleaned_text:
                    cleaned_doc = Document(page_content= cleaned_text,
                                           metadata= {**doc.metadata,
                                                      'cleaned': True,
                                                      'original_index': i})
                    cleaned_docs.append(cleaned_doc)

            except Exception as e:
                print(f"Error processing document {i}: {e}")
                continue

        return cleaned_docs
