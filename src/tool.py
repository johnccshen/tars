import pandas as pd
import openai, os, tiktoken, requests, time
from datetime import datetime, timedelta
from dotenv import load_dotenv

from langchain.llms.openai import OpenAI
from langchain.chat_models.openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()
openai.organization = os.environ['OPENAI_ORGANIZATION']
openai.api_key = os.environ['OPENAI_API_KEY']

class Tool():
    def __init__(self) -> None:
        # model/db we are using
        self.chat_model = "gpt-3.5-turbo-0613"
        self.llm_model = "text-davinci-003"
        self.embedding = "text-embedding-ada-002-v2"
        self.vectordb_place = "chroma"
        # model parameters
        self.max_tokens = 256
        self.temperature = 0
    
    def init_chat_model(self) -> ChatOpenAI:
        chat_model = ChatOpenAI(max_tokens=self.max_tokens,
                                temperature=self.temperature,
                                model=self.chat_model)
        return chat_model
    
    def init_llm_model(self) -> OpenAI:
        llm_model = OpenAI(max_tokens=self.max_tokens,
                           temperature=self.temperature,
                           model=self.llm_model)
        return llm_model
    
    def init_embedding(self) -> OpenAIEmbeddings:
        embedding = OpenAIEmbeddings()
        return embedding
    
    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    
    def check_index_exist(self) -> bool:
        if((os.path.isfile("faiss/index.faiss")) 
            and os.path.isfile("faiss/index.pkl")):
            return True
        else:
            return False

    def check_vectordb_exist(self) -> bool:
        if(os.path.exists(f"{self.vectordb_place}/chroma.sqlite3")):
            return True
        else:
            return False
        
    def check_usage(self) -> float:
        '''This function cannot use now, because we have request rate limit (5 times/min)'''
        df = pd.DataFrame()
        start_date = datetime(2023,12,11).date()

        while start_date <= datetime.today().date():
            # print("Process data on {} ...".format(start_date))
            url = "https://api.openai.com/v1/usage?date={}".format(start_date)
            headers = {"Authorization": f"Bearer {openai.api_key}"}
            response = requests.get(url, headers=headers).json()
            try:
                data = response["data"]
                # merge multiple day record into one dataframe
                df_tmp = pd.DataFrame(data)
                df = pd.concat([df, df_tmp], ignore_index=True)
            except KeyError:
                print("Rate_limit_exceeded")

            start_date += timedelta(days=1)
            time.sleep(3)

        context_token = df.groupby('snapshot_id')['n_context_tokens_total'].sum()
        generated_token = df.groupby('snapshot_id')['n_generated_tokens_total'].sum()

        input = pd.DataFrame({'token':context_token,'cost':context_token/1000})
        # price info is get from official website
        input.iloc[0,1] *= 0.0015
        input.iloc[1,1] *= 0.02
        input.iloc[2,1] *= 0.0001

        output = pd.DataFrame({'token':generated_token,'cost':generated_token/1000})
        # price info is get from official website
        output.iloc[0,1] *= 0.002
        output.iloc[1,1] *= 0.02
        output.iloc[2,1] *= 0.0001

        return round(sum(input['cost']) + sum(output['cost']),5)