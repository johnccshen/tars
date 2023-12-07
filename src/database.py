import chromadb
from chromadb import Settings
from chromadb.utils import embedding_functions
from llama_index.llms import LlamaCPP
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader, set_global_service_context
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

database_output_folder = 'chroma/'

class Database():
    '''Instantiate the chromadb client'''
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(settings=Settings(allow_reset=True), path=database_output_folder)
        self.collection = None
        # Check connection status
        # if(self.client.heartbeat != None):
        #     print("Connection Success")
        # else:
        #     print("Connection Failed")

    def embed_document(self, doc_name: str, llm: LlamaCPP, embed_model: HuggingFaceEmbeddings) -> VectorStoreIndex:
        # To satisfy the collection name regulation
        collection_name = str.lower(doc_name).replace(' ','_')
        huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
            api_key="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",
            model_name="BAAI/bge-large-en-v1.5"
        )
        # metadata is to setting distance function, embedding_function is to set our own embedding
        self.collection = self.client.get_or_create_collection(collection_name, 
                                                               metadata={"hnsw:space": "cosine"},
                                                               embedding_function=huggingface_ef)
        # Create a space in ChromaDB to store document embeddings
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        # Basic configuration on commonly used resources used during the indexing and querying stage
        service_context = ServiceContext.from_defaults(chunk_size=512,
                                                       llm=llm,embed_model=embed_model)
        set_global_service_context(service_context)

        if self.collection.count() < 1:
            #  Case1: create new collection and save document contents (vector type) to ChromaDB
            documents = SimpleDirectoryReader(
                    input_files=['doc/{}.pdf'.format(doc_name)]
                ).load_data()
            # Abstractions of storage vector
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # Building index based on provided document
            index = VectorStoreIndex.from_documents(
                    documents, storage_context=storage_context, show_progress=True
                )
            print("\n")
        else:
            # Case2: retrieve existing document contents (vector) from ChromaDB
            index = VectorStoreIndex.from_vector_store(
                vector_store,
            )
        return index
