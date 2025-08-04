import spacy
from langchain.schema import Document
from typing import List, Dict, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Constants defining max chunk size and overlap between chunks
MAX_TOKENS_FOR_CHUNKING = 350
OVERLAP_FOR_CHUNKING = 60

def process_document_batch_static(batch_data: Tuple, max_tokens: int, overlap: int) -> List[Document]:
    """
    Process a batch of documents in a subprocess to create chunked Document objects.

    This function is static and designed to be called by multiprocessing processes. It loads its own spaCy pipeline.
    """
    texts, metadatas, start_idx = batch_data
    chunks = []

    # Load spaCy model inside each subprocess
    nlp = spacy.load("en_core_web_sm", disable=[
        "ner", "parser", "textcat", "tok2vec", "attribute_ruler", "lemmatizer"
    ])
    nlp.add_pipe("sentencizer") # Add sentencizer for sentence boundary detection

    # Process all texts in the batch through spaCy pipeline
    for doc_idx, spacy_doc in enumerate(nlp.pipe(texts, n_process=1)):
        original_metadata = metadatas[doc_idx]
        processed_sentences = []

        for sent_idx, sent in enumerate(spacy_doc.sents):
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            # Token filtering: keep alphabetic tokens only, exclude stop words
            tokens = [t.text.lower() for t in sent if t.is_alpha and not t.is_stop]
            if tokens:
                # Construct cleaned sentence dict with text and token count
                processed_sentences.append({
                    'text': " ".join(tokens),
                    'token_count': len(tokens),
                    'original_text': sent_text,
                    'sent_idx': sent_idx
                })

        # If there are sentences, chunk them with overlap consideration
        if processed_sentences:
            doc_chunks = create_chunks_with_overlap(processed_sentences, max_tokens, overlap)

            for chunk_idx, chunk_data in enumerate(doc_chunks):
                chunk_metadata = {
                    **original_metadata,
                    'doc_index': start_idx + doc_idx,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(doc_chunks),
                    'sentence_range': chunk_data['sentence_range'],
                    'is_chunked': True,
                }

                chunks.append(Document(
                    page_content=chunk_data['text'],
                    metadata=chunk_metadata
                ))

    # Return all chunked Documents from this batch
    return chunks


def create_chunks_with_overlap(sentences: List[Dict], max_tokens: int, overlap: int) -> List[Dict]:
    """
    Combine sentences into chunks of text respecting the max tokens limit and overlap.
    """
    if not sentences:
        return []

    chunks = []
    i = 0

    # Iterate through all sentences building chunks based on token limits
    while i < len(sentences):
        chunk_sentences = []
        chunk_token_count = 0
        chunk_start_idx = i

        # Accumulate sentences until adding one more would exceed max_tokens
        while i < len(sentences):
            sent = sentences[i]
            if chunk_token_count + sent['token_count'] > max_tokens and chunk_sentences:
                break # Stop if adding sentence exceeds limit

            chunk_sentences.append(sent)
            chunk_token_count += sent['token_count']
            i += 1

        if chunk_sentences:
            chunk_text = " ".join(sent['text'] for sent in chunk_sentences)
            chunks.append({
                'text': chunk_text,
                'token_count': chunk_token_count,
                'sentence_range': (chunk_start_idx, i - 1),
                'sentence_count': len(chunk_sentences)
            })

            # Calculate how many sentences from the end to overlap for next chunk
            if i < len(sentences):
                overlap_tokens = overlap_sentences = 0
                for j in range(len(chunk_sentences) - 1, -1, -1):
                    sent_tokens = chunk_sentences[j]['token_count']
                    if overlap_tokens + sent_tokens <= overlap:
                        overlap_tokens += sent_tokens
                        overlap_sentences += 1
                    else:
                        break

                # Move back index i to include overlap sentences in next chunk
                i = max(chunk_start_idx + 1, i - overlap_sentences)

    return chunks


class ChunkSuperFast:
    """
    Class for efficiently chunking large numbers of Document objects in parallel using multiprocessing.

    It splits the document list into batches, processes each batch in parallel subprocesses,
    performs sentences segmentation, token filtering and chunks with overlap preserving context.
    """
    def __init__(self, docs: List[Document]):
        self.docs = docs
        self.max_tokens = MAX_TOKENS_FOR_CHUNKING
        self.overlap = OVERLAP_FOR_CHUNKING
        self.n_cores = min(mp.cpu_count(), 8) # Use maximum 8 cores for stability

    def chunk_pdfs_parallel(self, batch_size=100) -> List[Document]:
        # Chunk all documents in parallel batches using multiprocessing.
        if not self.docs:
            return []

        print(f"Processing {len(self.docs)} documents in batches of {batch_size}...")

        batches = []
        for i in range(0, len(self.docs), batch_size):
            batch_docs = self.docs[i:i + batch_size]
            texts = [doc.page_content for doc in batch_docs]
            metadatas = [doc.metadata for doc in batch_docs]
            batches.append((texts, metadatas, i))

        all_chunks = []

        # Parallelize chunking by submitting batches to process pool
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            future_to_batch = {
                executor.submit(process_document_batch_static, batch, self.max_tokens, self.overlap): batch
                for batch in batches
            }

            # As each batch finishes, gather the chunked documents
            for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Processing batches"):
                batch_chunks = future.result()
                all_chunks.extend(batch_chunks)

        print(f"Created {len(all_chunks)} chunks")
        return all_chunks
