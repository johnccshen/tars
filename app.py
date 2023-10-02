from llama_cpp import Llama
import chainlit as cl


# model_path = "models/llama/codellama-7b.Q4_K_M.gguf"
model_path = "models/llama/llama-2-13b-chat.Q4_K_M.gguf"


def run(prompt, model_path=model_path):
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        max_tokens=2048,
        temperature=0.1,
        # repeat_penalty=1.15,
        n_gpu_layers=1000,
    )

    
    output = llm(prompt, stop=["Question:", "\n"], echo=True)
    print(output)
    return output

@cl.on_message
async def main(message: str):
    # Your custom logic goes here...
    system_message = "\nI want you to pretend you are a AI doctor. Keep the responses less than 350 words. "
    prompt = f"Question: {message} Answer: "
    resp = run(system_message + prompt)
    # Send a response back to the user
    ans = resp.get("choices")[0].get("text")
    await cl.Message(
        content=ans.split("Answer: ")[1],
    ).send()
