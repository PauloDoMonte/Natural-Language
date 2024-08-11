from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
import uuid


chroma_client = chromadb.HttpClient(host="127.0.0.1", port = 8000, settings=Settings(allow_reset=True, anonymized_telemetry=False))

document_collection = chroma_client.get_or_create_collection(name="teste")


loader = TextLoader('text.txt', encoding='utf-8')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

for doc in docs:
    document_collection.add(
        ids=[str(uuid.uuid4())], metadatas=doc.metadata, documents=doc.page_content
    )

db4 = Chroma(
    client=chroma_client,
    collection_name="teste",
    embedding_function=embedding_function,
)

queries = ['O que é um banco de dados vetorial ?',
           'Tem algum exemplo de banco de dados vetorial ?','Como posso implementar um banco de dados vetorial ?']

for query in queries:
    
    docs = db4.similarity_search(query)
    print(docs[0].page_content)