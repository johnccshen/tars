import argparse, logging, os
from src.tool import Tool
from query_doc import query_document
from langchain.memory import ConversationSummaryBufferMemory

# Add logging
LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
DATE_FORMAT = '%Y%m%d %H:%M:%S'
logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename="log/system.log")

tool = Tool()
# Once the chat_history exceed max_token_limit, the llm will summarize the previous conversation.
memory = ConversationSummaryBufferMemory(llm=tool.init_chat_model(),
                                         max_token_limit=300, memory_key="chat_history",
                                         return_messages=True, output_key="answer")

if __name__ == "__main__":
    logging.info('System start')
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
        logging.info(f'User query: {_input}')
        if not _input:
            break
        print('\n')
        resp = query_document(_input, memory, args.verbose)
        logging.info(f'LLM response: {resp}')
        print(resp)
        print("\n")

    print("Close session! Thank you!")
    # Because the request rate limit, this function cannot use now
    # print("Process billing info ...\n")
    # cost = tool.check_usage()
    # print(f"Currently, you have to pay {cost} dollars on OpenAI API")
    # logging.info(f"OpenAI usage: {cost} dollars")
    logging.info("System exit")