# Import necessary libraries
from langchain_openai import OpenAIEmbeddings  # For using OpenAI's embedding models through LangChain
from dotenv import load_dotenv  # For loading environment variables from a .env file

# Load environment variables (like your OpenAI API key) from a .env file
load_dotenv()

# Instantiate the OpenAIEmbeddings object
# - model: specifies which embedding model to use
# - dimensions: downsizes the output vector to 32 dimensions (for faster processing / lighter storage, at the cost of detail)
embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

# List of sample documents to embed (each will be converted into a numerical vector representation)
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

# Generate embeddings for each document
# This converts each string into a 32-dimensional vector capturing semantic meaning
result = embedding.embed_documents(documents)

# Output the resulting embeddings to the console
# (Each document gets a separate vector — useful for semantic search, clustering, etc.)
print(str(result))
