# Import necessary libraries
import os  # For setting environment variables
from langchain_huggingface import HuggingFaceEmbeddings  # LangChain wrapper for Hugging Face embeddings

# Set a custom cache directory for Hugging Face models
# This is useful for managing model storage locations, especially on Windows or non-default drives
os.environ['HF_HOME'] = 'D:/HF/Sentence transformer'

# Initialize the Hugging Face embedding model
# - model_name: Specifies the pretrained sentence transformer to use
#   'all-MiniLM-L6-v2' is small, fast, and decent for most semantic tasks
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# List of example documents (strings) to embed
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

# Generate embeddings for each document
# This returns a list of high-dimensional vectors (dense embeddings)
vector = embedding.embed_documents(documents)

# Print the resulting vectors
# Each document is represented as a numerical vector capturing its semantic meaning
print(str(vector))
