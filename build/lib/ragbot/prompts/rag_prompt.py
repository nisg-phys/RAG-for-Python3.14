from langchain_core.prompts import ChatPromptTemplate


RAG_PROMPT_TEMPLATE = """
You are a helpful assistant well versed in government policies and schemes. 
Answer the question ONLY using the context below. If you do not know the answer refuse politely.

<context>
{context}
</context>

Question: {question}
"""


rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)