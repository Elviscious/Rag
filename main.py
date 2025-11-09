import asyncio
from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os


load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature="0.5",
    max_output_tokens=200,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    collection_name="rag_docs",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


system_prompt = """
You are a helpful AI assistant that helps people find information.
Answer the questions truthfully and strictly based on the context given.
**IMPORTANT:** If the answer is not found in the context provided, you MUST reply with the phrase: 'I cannot answer that question as the relevant information is not in my sources.'
Avoid adding unrelated or made-up information.
"""

template = """
{system_prompt}

Context:
{context}


Question: 
{question}

Answer:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["system_prompt", "context", "question"]
)

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever,
#     chain_type_kwargs={"prompt": prompt},
#     return_source_documents=True
# )


def ask_rag(question):
    docs = retriever.invoke(question)

    print("--- DEBUG: RETRIEVED DOCS ---")
    print(docs)
    print("------------------------------")

    context = "\n\n".join([d.page_content for d in docs])
    final_prompt = prompt.format(
        system_prompt=system_prompt,
        context=context,
        question=question
    )
    response = llm.invoke(final_prompt)

    return {
        "answer": response.content,
        "source_documents": docs
    }


question = "What is Levenshtein distance?"
result = ask_rag(question)

print("Answer:\n", result["answer"])
print("\nSources:")
unique_sources = set()

for doc in result["source_documents"]:
    source = doc.metadata.get("source", "unknown")

    unique_sources.add(source)

for src in unique_sources:
    print("-", src)
