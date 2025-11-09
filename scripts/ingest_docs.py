from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import os, json
import chromadb
from chromadb.utils import embedding_functions




def load_all_pdfs(data_dir):
    all_docs = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_dir, filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs

documents = load_all_pdfs("data")
print(f"Loaded {len(documents)} documents.")


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = [chunk.page_content for chunk in chunks]
metadata = [chunk.metadata for chunk in chunks]

embeddings = model.encode(texts).tolist()



chroma_client = chromadb.PersistentClient(path="chroma_db")

collection = chroma_client.get_or_create_collection(name="rag_docs")


collection.add(
    documents=texts,
    embeddings=embeddings,
    metadatas=metadata,
    ids = [f"doc_{i}" for i in range(len(texts))],
)

print("saved to chroma DB.")