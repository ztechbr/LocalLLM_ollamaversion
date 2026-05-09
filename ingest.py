
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DOCS_PATH = "./docs"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

documents = []

for pdf in Path(DOCS_PATH).glob("*.pdf"):
    print(f"Processando: {pdf}")

    loader = PyPDFLoader(str(pdf))
    pages = loader.load()

    chunks = splitter.split_documents(pages)

    documents.extend(chunks)

db = FAISS.from_documents(documents, embeddings)

db.save_local("./rag_store")

print("RAG FAISS criado com sucesso.")
