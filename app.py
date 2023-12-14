import argparse
from src.tool import Tool
from query_doc import query_document
from langchain.memory import ConversationSummaryBufferMemory

tool = Tool()
# not working
memory = ConversationSummaryBufferMemory(llm=tool.init_chat_model(),
                                         max_token_limit=1024, memory_key="chat_history",
                                         return_messages=True, output_key="answer")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-vv", "--verbose", type=bool, default=False, action=argparse.BooleanOptionalAction,
                         help="show verbose message about llm chain")
    args = parser.parse_args()

    print("\n")
    print("Hint: Type Enter to exit!")
    print(f"Hint: Use model: {tool.llm_model}")
    print(f"Hint: Use embedding: {tool.embedding}")
    print("\n")

    while True:
        print("Enter your question here:")
        _input = input()
        if not _input:
            break
        print('\n')
        resp = query_document(_input, memory, args.verbose)
        print(resp)
        print("\n")

    print("Close session! Thank you!")
    print(f"Currently, you have to pay {tool.check_usage()} dollars on OpenAI API")
    print('\n')
    # print(memory.load_memory_variables({}))
