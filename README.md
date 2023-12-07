# Tars 

### Prerequisite
- You need to prepare a Python environment and install poetry in your computer. ([Poetry Document](https://python-poetry.org/docs/))
- Use `poetry install` command to install package dependency
- You can choose either download the following model and put them under `/model/{model_name}` folder, or uncomment download command in `query_doc.py`

### Model
- LLM model: [TheBloke/Mistral-7B-OpenOrca-GGUF](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/blob/main/mistral-7b-openorca.Q5_K_S.gguf)
- Embedding model: [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5/tree/main)

### Usage
- You need to use confluence.py to download Confluence page.
```
python confluence.py -l configuration_guideline
```
- Execute app.py to query information from document. If you want to turn on verbose message use the following command, else just type `python app.py`.
```
python app.py --verbose
```

### Example
```
(tars-py3.11) sean@1302096-NB:$ python app.py

Hint: Type Enter to exit!
Hint: Use model: models/TheBloke_Mistral-7B-OpenOrca-GGUF/mistral-7b-openorca.Q5_K_S.gguf

Enter your question here:
Briefly list all branch name and what they are processed in traumatic spine injury cycles

 The branches processed in Traumatic Spine Injury cycles include:
1. Spinal Cord labeling (B1) - White & Grey Matter Segmentation
2. Cop Task Branch (B2) - Copies offline exams to their original versions
3. Segmentation Review Branch (B2) - Reviews segmentation results and edits if needed
4. Main Task (B3) - Registration, resampling, and C3D processing between different spaces
5. Boundary Shift Integral computation (B4) - Computes BSI scores for atrophy assessment
6. Final Review of Hyperintensity Ratio (B5) - Compares T2* labeled cord and MTR exam types with BSI scores
```