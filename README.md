# Tars 
> A chatbot that use OpenAI model to answer question about Confluence document

### Setting
- You need to prepare a Python environment and install poetry in your computer. ([Poetry Document](https://python-poetry.org/docs/))
- Use `poetry install` command to install package dependency. For your convinence, I recommend you launch this project in a **Linux** environment.
- Create `system.log` file under `log` folder to store log and a `.env` file under currnt folder, please make sure use below format to fill your OpenAI token in `.env` file.
```
OPENAI_ORGANIZATION='Your_OPENAI_Token'
OPENAI_API_KEY='Your_OPENAI_Token'
```
- The `src/confluence_loader.py` is the example code about how to use `ConfluenceLoader` components provided by `langchain`, which can help you download data directly from Confluence. However, it will combine all data into single Document class, so I **didn't recommend** it.

### Model / VectorDB
- LLM model: `text-davinci-003` provided by OpenAI
- Embedding model: `text-embedding-ada-002-v2` provided by OpenAI
- Vector DB: 
    - `Chroma` provided by Chroma
    - `FAISS` provided by Meta

### Prerequisites
1. Currently, the `check_usage` function is broken, because the OpenAI API have **request rate limit** (5 requests / min), so we cannot retrieve total billing info.
2. In the `src/database.py`, I leave two solutions which is about how to store document as a chunk. Solution 1 is about only extract text from relevant data in a file to a single chunk, and store them into FAISS. However, the second method is to keep data as element type and store relevant data in a Chroma collection, and separate relevant data from different file into different database.
3. Because the internal retriever provided by langchain does not return similarity score on each Document, so we creat new retriever called `MyVectorStoreRetriever` under `model/my_retriever.py` to replace original `vectorstore.as_retriever()` method.

### Usage
- You need to use confluence.py to download Confluence page.
```
python confluence.py -l configuration_guideline
```
- Execute app.py to query information from document. If you want to turn on verbose message use the following command, else just type `python app.py`.
```
python app.py --verbose
```
- Once the program start executing, you can input the question you want to ask.

### Example
```
(tars-py3.11) sean@1302096-NB:$ python app.py

Hint: Type Enter to exit!
Hint: Use model: text-davinci-003
Hint: Use embedding: text-embedding-ada-002-v2

Enter your question here:
what's the input exam name in Infant FS Analysis cycle

The exams needed for Infant FS are 3D T1-weighted, 3D T1 Bias Field Corrected, Segmentation Mask, ASEG_MASK, and Brain Mask. Each exam must have Anatomy code BRAIN, Modality code MR, and Directionality unspecified. Only 4 exams have Laterality specified, and Age In Month must be inserted in the QC session.

Close session! Thank you!
```