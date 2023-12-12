import warnings, time
warnings.filterwarnings('ignore')

# Import the llama index HF Wrapper
from llama_index.llms import HuggingFaceLLM
# Import the prompt wrapper...but for llama index
from llama_index.prompts.prompts import SimpleInputPrompt
from ctransformers import AutoModelForCausalLM, AutoConfig
from transformers import AutoTokenizer
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
import transformers, torch

from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from langchain.embeddings import HuggingFaceEmbeddings
from src.database import Database

doc_name = "Traumatic Spine Injury"

def get_tokenizer_llm() -> transformers.pipeline:
    '''Load LLM model and their tokenizer from the model folder'''

    # If you occur Number of tokens (xxxx) exceeded maximum context length(512) error, remember to set config as below.
    config = AutoConfig.from_pretrained("models/TheBloke_Mistral-7B-OpenOrca-GGUF")
    config.config.max_new_tokens = 2048
    config.config.context_length = 4096
    # Generic way to load model from HuggingFace
    model = AutoModelForCausalLM.from_pretrained("models/TheBloke_Mistral-7B-OpenOrca-GGUF", 
                                           model_file="mistral-7b-openorca.Q5_K_S.gguf", 
                                           model_type="mistral", hf=True, config=config)

    tokenizer = AutoTokenizer.from_pretrained(
        "models/TheBloke_Mistral-7B-OpenOrca-GGUF/tokenizer/"
    )

    # First time use, you need to store tokenizer in your local device
    # so please remember to uncomment below code

    # tokenizer = AutoTokenizer.from_pretrained(
    #     "Open-Orca/Mistral-7B-OpenOrca"
    # )
    # tokenizer.save_pretrained("models/TheBloke_TinyLlama-1.1B-Chat-v0.3-GGUF/tokenizer/")

    pipeline = transformers.pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        return_full_text = True, temperature=0.1, repetition_penalty=1.1
    )
    return pipeline

def query_document(prompt: str) -> str:
    '''It will receive query message from app.py and return processed response'''

    start_time = time.time()
    # The original LLM initialize method has already been removed
    # you can get it from dev branch
    generator = get_tokenizer_llm()
    llm = HuggingFacePipeline(pipeline=generator)

    # Often costs around 200 sec.
    print("load llm & tokenizer cost: {}".format(str(time.time() - start_time)))

    # It will check your model folder, if the model doesn't exist, it will download from the Internet
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5", cache_folder="models")

    chain = Database().embed_document_langchain(doc_name, llm, embed_model)
    
    print("start query!")
    start_time = time.time()
    # testing the model (RetrievalQA)
    result = chain({"query": prompt})
    print("retrieve data & generate response: {}".format(str(time.time() - start_time)))
    print('\n')
    print('\n')
    return result["result"]

    # qa = Database().embed_document_langchain(doc_name, llm, embed_model)
    # testing the model (ConversationalRetrievalChain)
    # result = qa({"question": prompt, 'chat_history': []})
    # return result['answer']

    # Old method by using index to query relevant data
    # index = Database().embed_document(doc_name, llm, embed_model)
    # set up query engine, can change to chat_engine or other else
    # query_engine = index.as_query_engine()
    # retrieve relevant information and generate response through LLM
    # response = query_engine.query(prompt)
    # return response
    