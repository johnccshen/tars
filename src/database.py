import chromadb
from src.tool import Tool
from model.my_retriever import MyVectorStoreRetriever
from langchain.docstore.document import Document
from langchain.llms.openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain

# If you want to debug, please set threshold as 0.0
similarity_threshold = 0.65

# stringify list element into string
def listToString(s):
  str1 = " "
  return str1.join(s)

# elem needs to be langchain.Document class
def get_key(elem: Document) -> [int, str, str]:
  depth = elem.metadata.get('category_depth', None)
  parent_id = elem.metadata.get('parent_id', None)
  category = elem.metadata['category']

  return depth, parent_id, category

def embed_document(llm: OpenAI, embedding: OpenAIEmbeddings, memory: ConversationSummaryBufferMemory, verbose: bool) -> BaseConversationalRetrievalChain:  
    '''Use ConversationalRetrievalChain to generate response from user's query'''
    tool  = Tool()
    # Check ChromaDB file exists or not, if you change vectordb to faiss you need to change the check document
    if(not tool.check_vectordb_exist()):
        # Extract information from html file
        docs = UnstructuredHTMLLoader("doc/Infant Studies.html", mode="elements").load()
        
        # Combine same section elements (under same H2) into a new chunks
        index = 0
        chunk_list = []
        while index < len(docs):
            chunks = []
            depth, parent_id, category = get_key(docs[index])

            # If we meet H2 element, append into list first
            if(depth==1 and category=="Title" and parent_id!=None):
                # check metadata contains list data or not, if yes stringify them
                for key in docs[index].metadata.keys():
                    if isinstance(docs[index].metadata[key], list):
                        str_list = listToString(docs[index].metadata[key])
                        docs[index].metadata[key] = str_list
                chunks.append(docs[index])
                index += 1

            while index < len(docs):
                depth, parent_id, category = get_key(docs[index])
                # If we meet H2 element, break current chunks
                if(depth==1 and category=="Title" and parent_id!=None):
                    break
                else:
                    # check metadata contains list data or not, if yes stringify them
                    for key in docs[index].metadata.keys():
                        if isinstance(docs[index].metadata[key], list):
                            str_list = listToString(docs[index].metadata[key])
                            docs[index].metadata[key] = str_list
                    chunks.append(docs[index])
                    index += 1
    
            chunk_list.append(chunks)
        #-------------------------------------------------------------
        # Sol1: Combine same section's page content into one chunk
        documents = []
        for doc in chunk_list:
            # Combine page_content
            content = ""
            for elem in doc:
                content += elem.page_content
                content += '\n'
            # Wrapped data with Document class
            tmp_doc = Document(
                page_content = content,
                metadata = {
                    'source': "doc/Infant Studies.html",
                    'filetype': 'text/html',
                    'category': doc[0].page_content # h2 title
                }
            )
            documents.append(tmp_doc)
        # save embeddings to disk
        vectorstore = FAISS.from_documents(documents=documents, embedding=embedding)
        vectorstore.save_local("faiss")
        #-------------------------------------------------------------
        # Sol2: Dump one section's data into one collection, one file one database
        # Initialize chromadb client
        # num = 1
        # persistent_client = chromadb.PersistentClient(path=tool.vectordb_place)
        
        # for col in chunk_list:
        #     # In order to satisfy the collection name rule, change the name
        #     final_name = "infant_"+str(num)

        #     vectordb = Chroma.from_documents(
        #         col, embedding, client=persistent_client,
        #         collection_name=final_name, persist_directory=tool.vectordb_place
        #     )
        #     # save embeddings to disk
        #     vectordb.persist()
        #     num += 1
        #---------------------------------------------------------------
    
    # load embeddings from disk (solution 1)
    vectorstore = FAISS.load_local("faiss", embedding)
    
    # load embeddings from disk (solution 2) 
    # You need to define collection name before initialize Chroma client, or use abstract to determine load which document
    # vectorstore = Chroma(collection_name="infant_3", embedding_function=embedding, persist_directory=tool.vectordb_place)
    retriever = MyVectorStoreRetriever(
        vectorstore=vectorstore,
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": similarity_threshold, "k": 1},
    )

    # default retriever will not return similarity score, but keep it for in case
    # retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":4}) 

    # define separately about question-generating & question-answering chain
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    # map_reduce type is good at dealing with multiple documents
    doc_chain = load_qa_chain(llm, chain_type="map_reduce")
    # setting chain to return generated_question & source documents it retrieve from Chroma
    chain = ConversationalRetrievalChain(retriever=retriever,memory=memory,
                                         combine_docs_chain=doc_chain, verbose=verbose,
                                         question_generator=question_generator,
                                         return_generated_question=True,
                                         return_source_documents=True)
    return chain