# Import necessary libraries
import os
from langchain_huggingface import HuggingFaceEmbeddings  # LangChain wrapper for Hugging Face embeddings
from dotenv import load_dotenv  # For loading environment variables (like API keys) from a .env file
from sklearn.metrics.pairwise import cosine_similarity  # For calculating similarity between vectors
import numpy as np  # For numerical operations

# Load environment variables (ensure your .env file contains your OpenAI API key)
load_dotenv()
# Set a custom cache directory for Hugging Face models
# This is useful for managing model storage locations, especially on Windows or non-default drives
os.environ['HF_HOME'] = 'D:/HF/Sentence transformer'

# Initialize OpenAI embedding model
# - model: specifies which OpenAI embedding model to use
# - dimensions: reduces the dimensionality to 300 for efficiency while retaining semantic strength
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Define a list of cricket-related documents
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# Define a user query to match against the documents
query = 'tell me about bumrah'

# Generate embeddings for all documents
doc_embeddings = embedding.embed_documents(documents)

# print(doc_embeddings) # This will print the document embeddings after they are generated

# Generate an embedding for the query
query_embedding = embedding.embed_query(query)

# print(query_embedding) # This will print the query embedding after it is generated

# Compute cosine similarity between the query vector and each document vector
# Returns a list of similarity scores (one per document)
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Find the index and score of the most similar document
# - enumerate pairs each score with its index
# - sorted keeps them in ascending order
# - [-1] grabs the highest scoring (most similar) document
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

# Output the original query
print('Question: ',query)

# Output the document most similar to the query
print('Answer: ',documents[index])

# Print the similarity score to see how strong the match is
print("similarity score is:", score)
