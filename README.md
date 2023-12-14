# Tars 

### Prerequisite
- You need to prepare a Python environment and install poetry in your computer. ([Poetry Document](https://python-poetry.org/docs/))
- Use `poetry install` command to install package dependency

### Model / VectorDB
- LLM model: `text-davinci-003` provided by OpenAI
- Embedding model: `text-embedding-ada-002-v2` provided by OpenAI
- Vector DB: `Faiss` provided by Meta

### Usage
- You need to use confluence.py to download Confluence page.
```
python confluence.py -l configuration_guideline
```
- Execute app.py to query information from document. If you want to turn on verbose message use the following command, else just type `python app.py`.
```
python app.py --verbose
```
- Once the program start executing, you can input the question you want to ask. When the program is stop, it will show the billing info about API usage.

### Example
```
(tars-py3.11) sean@1302096-NB:$ python app.py

Hint: Type Enter to exit!
Hint: Use model: text-davinci-003
Hint: Use embedding: text-embedding-ada-002-v2

Enter your question here:
what's the input exam name in Infant FS Analysis cycle

The input exam name in Infant FS Analysis cycle is ASEG_MASK, 3D_T1_Bias_Field_Corrected.

Close session! Thank you!
Currently, you have to pay 1.03793 dollars on OpenAI API
```