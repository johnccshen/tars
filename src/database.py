from src.tool import Tool
from langchain.llms.openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain

def embed_document(llm: OpenAI, embedding: OpenAIEmbeddings, memory: ConversationSummaryBufferMemory, verbose: bool) -> BaseConversationalRetrievalChain:  
    '''Use ConversationalRetrievalChain to generate response from user's query'''
    tool  = Tool()
    # Check FAISS index file exists or not
    if(not tool.check_index_exist):
        # Extract information from markdown file
        documents = UnstructuredMarkdownLoader("doc/Infant Studies.md", mode="elements").load()
        
        # save embeddings to disk (remember to comment below code, otherwise OpenAI maybe embedded again, not sure)
        vectorstore = FAISS.from_documents(documents=documents, embedding=embedding)
        vectorstore.save_local(tool.database_place)

    # load embeddings from disk
    vectorstore = FAISS.load_local(tool.database_place, embedding)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":4}) 

    # define separately about question-generating & question-answering chain
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(llm, chain_type="map_reduce")
    # setting chain to return generated_question & source documents it retrieve from FAISS
    chain = ConversationalRetrievalChain(retriever=retriever,memory=memory,
                                         combine_docs_chain=doc_chain, verbose=verbose,
                                         question_generator=question_generator,
                                         return_generated_question=True,
                                         return_source_documents=True)
    return chain