from langchain.document_loaders import ConfluenceLoader
import os
from dotenv import load_dotenv

load_dotenv()
# You need to create a new personal access token on settings
confluence_token = os.environ["CONFLUENCE_TOKEN"]

loader = ConfluenceLoader(
    url="https://confluence.imgdev.bioclinica.com/",
    token=confluence_token
)
# Can load documents from Confluence directly, but it combine all elements together into same Document class
documents = loader.load(label="configuration_guideline", include_attachments=False, limit=50)
print(len(documents))
print(documents)