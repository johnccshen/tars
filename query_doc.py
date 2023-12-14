import warnings
warnings.filterwarnings('ignore')
from src.tool import Tool
from src.database import embed_document

from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationSummaryBufferMemory

def query_document(prompt: str, memory: ConversationSummaryBufferMemory, verbose: bool) -> str:
    '''It will receive query message from app.py and return processed response'''
    tool = Tool()
    prompt_token = tool.num_tokens_from_string(prompt, tool.llm_model)
    print(f"Prompt Tokens: {prompt_token} (charged by embedding model price)")

    llm = tool.init_llm_model()
    embedding = tool.init_embedding()
    chain = embed_document(llm, embedding, memory, verbose)
    
    # testing the model (ConversationalRetrievalChain), and use tools to get additional info
    with get_openai_callback() as cb:
        result = chain({"question": prompt})
        print(f"total tokens: {cb.total_tokens}")
        print(f"prompt tokens: {cb.prompt_tokens}")
        print(f"completion tokens: {cb.completion_tokens}")
        print(f"cost is: {cb.total_cost}")
    
    print('\n')
    print("Generated question:")
    print(result["generated_question"])
    print('\n')

    print("First source documents:")
    print(result['source_documents'][0])
    print('\n')

    print("Summary of chat_history:")
    print(memory.buffer())
    print('\n')
    return result['answer']
    