import warnings
warnings.filterwarnings('ignore')
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from langchain.embeddings import HuggingFaceEmbeddings

def query_document(prompt: str, verbose: bool) -> str:
    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    documents = SimpleDirectoryReader(
            input_files=['doc/Traumatic Spine Injury.pdf']
        ).load_data()
    # load llama model file and embed model
    llm = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        # model_url='https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q5_K_S.gguf?download=true'
        model_url=None,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path='models/TheBloke_Mistral-7B-OpenOrca-GGUF/mistral-7b-openorca.Q5_K_S.gguf',
        temperature=0.1,
        max_new_tokens=256,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=2048,
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": -1},
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=verbose,
    )
    # Load embed model from Internet
    # embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    # Load embed model from local
    embed_model = HuggingFaceEmbeddings(model_name="models/BAAI_bge-large-en-v1.5")

    # create a service context
    service_context = ServiceContext.from_defaults(chunk_size=1024, chunk_overlap=64,
                                                llm=llm,embed_model=embed_model)
    # create vector store index
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)

    # set up query engine
    query_engine = index.as_query_engine()
    response = query_engine.query(prompt)
    return response