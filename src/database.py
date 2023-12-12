import chromadb
from chromadb import Settings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from llama_index.llms import LlamaCPP
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader

database_output_folder = 'chroma/'
# Can try different kinds of template
template = """
        Given this text extracts:
        -----
        {context}
        -----
        Please answer with to the following question:
        Question: {question}
        Helpful Answer:
        """

class Database():
    def __init__(self) -> None:
        '''Instantiate the chromadb client'''
        self.client = chromadb.PersistentClient(settings=Settings(allow_reset=True), path=database_output_folder)
        self.collection = None
        # Check database connection status, leave it just for in case
        # if(self.client.heartbeat != None):
        #     print("Connection Success")
        # else:
        #     print("Connection Failed")

    def embed_document(self, doc_name: str, llm: LlamaCPP, embed_model: HuggingFaceEmbeddings) -> VectorStoreIndex:
        '''Reference: https://docs.llamaindex.ai/en/stable/examples/vector_stores/ChromaIndexDemo.html'''
        # To satisfy the collection name regulation
        collection_name = str.lower(doc_name).replace(' ','_')
        # metadata is to setting distance function, determine the similarity between query and chunks
        self.collection = self.client.get_or_create_collection(collection_name, 
                                                               metadata={"hnsw:space": "cosine"})
        # Create a space in ChromaDB to store document embeddings
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        # Basic configuration on commonly used resources used during the indexing and querying stage
        service_context = ServiceContext.from_defaults(chunk_size=512,
                                                       llm=llm,embed_model=embed_model)

        if self.collection.count() < 1:
            #  Case1: create new collection and save document contents (vector type) to ChromaDB
            documents = SimpleDirectoryReader(
                    input_files=['doc/{}.pdf'.format(doc_name)]
                ).load_data()
            # Abstractions of storage vector
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # Building index based on provided document
            index = VectorStoreIndex.from_documents(
                    documents, storage_context=storage_context,
                    service_context=service_context, show_progress=True
                )
            print("\n")
        else:
            # Case2: retrieve existing document contents (vector) from ChromaDB
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                service_context=service_context
            )
        return index
    
    def embed_document_langchain(self, doc_name: str, llm: HuggingFacePipeline, embed_model: HuggingFaceEmbeddings) -> BaseRetrievalQA:  
        '''In order to feed context to llm loaded by Auto method, I decided to use RetrievalQA'''

        # The default splitting method is divide documents by pages
        docs = PyPDFLoader("doc/{}.pdf".format(doc_name)).load_and_split()
        # Have tried to use RecursiveCharacterTextSplitter to split document into smaller chunks
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)                                                     
        # documents = text_splitter.split_documents(docs)

        # Wrapped around vector database, used for storing and querying embeddings
        vectorstore = Chroma.from_documents(docs, embed_model)
        # choose top 3 similar result from chunks
        retriever = vectorstore.as_retriever() 

        llama_prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        # llm need to be a pipeline
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                            chain_type_kwargs={"prompt": llama_prompt}, 
                                            return_source_documents=True)
        return chain
        # same as RetrievalQA but adds the memory to the chain, so it support multi-turn conversations
        # qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
        # return qa