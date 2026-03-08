# vector DB: Qdrant[lIGHTWEIGHT],Chroma,Weaviate,Redis,Milvus,Pinecone,Typesense,Vald,PGVector,ElasticSearch,Opensearch
# Langchain: LangSmith,LangServe,LangGraph,LangAgent,LangTool,LangMemory,LangPrompt,LangChainHub
# Langchain provide utlity tools to load documents, split them into smaller chunks, create vector embeddings and store them in vector databases. In this example we will use Qdrant as our vector database and OpenAI's text-embedding-3-large model to create vector embeddings of the documents. We will also use PyPDFLoader to load a PDF document and RecursiveCharacterTextSplitter to split the document into smaller chunks.
# Cammand to install langchain
# pip install -qU langchain-community pypdf
# pip install -qU langchain-openai langchain-qdrant langchain-text-splitters



from dotenv import load_dotenv

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

pdf_path = Path(__file__).parent / "nodejs.pdf"

# Load this file in python program
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

# Split the docs into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)

chunks = text_splitter.split_documents(documents=docs)

# Vector Embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="Rag"
)

print("Indexing of documents done....")