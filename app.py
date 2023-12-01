import argparse
from llama_cpp import Llama

MISTRAL_7B_OPENORCA_Q5 = '/data/llm/mistral-7b-openorca.Q5_K_S.gguf'


# following message comes from https://github.com/fastai/lm-hackers/blob/main/lm-hackers.ipynb
system_msg = """
You are an autoregressive language model that has been fine-tuned with instruction-tuning and RLHF. 
You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning. 
If you think there might not be a correct answer, you say so.

Since you are autoregressive, each token you produce is another opportunity to use computation, 
therefore you always spend a few sentences explaining background context, assumptions, 
and step-by-step thinking BEFORE you try to answer a question. However: if the request begins with the string "vv" 
then ignore the previous sentence and instead make your response as concise as possible, with no introduction or background at the start,
no summary at the end, and outputting only code for answers where code is appropriate.

Your users are experts in AI and ethics, so they already know you're a language model and your capabilities and limitations, 
so don't remind them of that. They're familiar with ethical issues in general so you don't need to remind them about those either. 
Don't be verbose in your answers, but do provide details and examples where it might help the explanation. When showing Python code, 
minimise vertical space, and do not include comments or docstrings; you do not need to follow PEP8, 
since your users' organizations do not do so.
"""
# system_msg = ""




def run(prompt, model, verbose):
    llm = Llama(model_path=model, verbose=verbose)
    print("asking...")
    outputs = llm(prompt, max_tokens=1024, stop=["Q:", "\n"], echo=True)
    return outputs

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default=MISTRAL_7B_OPENORCA_Q5)
    parser.add_argument("-vv", "--verbose", type=bool, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    
    query_template = "Question: {query} Answer: "
    query = system_msg
    print("\n")
    print("Hint: Type Enter to exit!")
    print(f"Hint: Use model: {args.model}")
    print("\n")

    while True:
        print("Enter your question here:")
        template = query_template
        _input = input()
        query += template.replace("{query}", _input)
        if not _input:
            break
        resp = run(query, args.model, args.verbose)
        print(resp["choices"][0].get("text"))
        print("\n")

    print("Close session! Thank you!")
