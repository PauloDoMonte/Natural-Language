import uuid
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

def connect(host: str, port: int) -> chromadb.HttpClient:
    """
    Conecta-se a um servidor ChromaDB e retorna um cliente HTTP.

    Args:
        host (str): O endereço do host onde o ChromaDB está sendo executado.
        port (int): A porta na qual o ChromaDB está sendo executado.

    Returns:
        chromadb.HttpClient: Um cliente HTTP configurado para interagir com o ChromaDB.
    """
    client = chromadb.HttpClient(host=host, port=port, 
                                 settings=Settings(allow_reset=True, anonymized_telemetry=False))
    return client

def add_file(host: str, port: int, file: str) -> Chroma:
    """
    Adiciona o conteúdo de um arquivo de texto a uma coleção no ChromaDB. Se a coleção
    correspondente ao nome do arquivo não existir, ela será criada.

    Args:
        host (str): O endereço do host onde o ChromaDB está sendo executado.
        port (int): A porta na qual o ChromaDB está sendo executado.
        file (str): O caminho para o arquivo de texto a ser adicionado ao ChromaDB.

    Returns:
        Chroma: Uma instância do Chroma configurada com a coleção correspondente ao arquivo.
    """
    client = connect(host, port)
    
    collection_name = file
    existing_collections = client.list_collections()
    
    collection_exists = any(collection['name'] == collection_name for collection in existing_collections)
    
    if not collection_exists:
        document_collection = client.create_collection(name=collection_name)
        
        loader = TextLoader(file, encoding='utf-8')
        documents = loader.load()
        
        text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        for doc in docs:
            document_collection.add(
                ids=[str(uuid.uuid4())], metadatas=doc.metadata, documents=doc.page_content
            )
            
    db4 = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_function
    )
    
    return db4
