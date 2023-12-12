import argparse
from query_doc import query_document

MISTRAL_7B_OPENORCA_Q5 = 'models/TheBloke_Mistral-7B-OpenOrca-GGUF/mistral-7b-openorca.Q5_K_S.gguf'
BGE_LARGE_EN_V1_5 = 'models/BAAI_bge-large-en-v1.5'

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default=MISTRAL_7B_OPENORCA_Q5)
    parser.add_argument("-e", "--embedding", type=str, default=BGE_LARGE_EN_V1_5)
    parser.add_argument("-vv", "--verbose", type=bool, default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    print("\n")
    print("Hint: Type Enter to exit!")
    print(f"Hint: Use model: {args.model}")
    print(f"Hint: Use embedding: {args.embedding}")
    print("\n")

    while True:
        print("Enter your question here:")
        # Example question: Briefly list all branch name and what they are processed in traumatic spine injury cycles
        _input = input()
        if not _input:
            break
        print('\n')
        resp = query_document(_input)
        print(resp)
        print("\n")

    print("Close session! Thank you!")
