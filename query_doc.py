import warnings, logging
warnings.filterwarnings('ignore')
from src.tool import Tool
from src.database import embed_document

from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationSummaryBufferMemory

def query_document(prompt: str, memory: ConversationSummaryBufferMemory, verbose: bool) -> str:
    '''It will receive query message from app.py and return processed response'''
    tool = Tool()
    llm = tool.init_llm_model()
    embedding = tool.init_embedding()

    # Test what chunks retriever send to LLM
    # retriever = embed_document(llm, embedding, memory, verbose)
    # response = retriever.invoke(prompt)
    # for text in response:
    #     print("---------------------")
    #     print(text.metadata['score'])
    #     print(text.metadata['category'])
    #     print(text.page_content)
    #     print("---------------------")
    #     print('\n\n')
    # return "done"
    
    # testing the model (ConversationalRetrievalChain), and use tools to get additional info
    chain = embed_document(llm, embedding, memory, verbose)
    with get_openai_callback() as cb:
        result = chain({"question": prompt})
        # Didn't reflect the real billing info
        # print(f"total tokens: {cb.total_tokens}")
        # print(f"prompt tokens: {cb.prompt_tokens}")
        # print(f"completion tokens: {cb.completion_tokens}")
        # print(f"cost is: {cb.total_cost}")
    
    # Can output some additional info to debug the Conversational chain
    # print('\n')
    # print("Generated question:")
    # print(result["generated_question"])
    # logging.info(f"Generated question: {result['generated_question']}")
    # print('\n')

    # print("Source documents:")
    # print(result["source_documents"])
    # logging.info(f"Source documents: {result['source_documents']}")
    # print('\n')

    return result['answer']
    