from langchain.prompts import PromptTemplate

"""
Defines a custom prompt template for the RAG application's LLM chain.
This prompt instructs the LLM to:
 - Use only the provided context when answering.
 - Answer truthfully, concisely, accurately, and use pointers.
 - If the context is empty or irrelevant, reply with a fixed message with a breif reason.
 - BUT: If the query is a basic conversational message, answer naturally.
 - All question and answer formatting is handled by this template.
"""
CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant with access to the following context:

{context}

Answer the question truthfully, concisely, accurately, and using pointers.
DISCLAIMER: If the question is not relevant to the context, or the context is empty, reply exactly:
"No relevant texts regarding query found." and add a short reason.
BUT: Respond to basic conversational questions.
eg. "Hi there", "How are you", "What are you doing", etc.

Question: {question}
Answer:
"""
)