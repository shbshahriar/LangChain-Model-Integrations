# Import the required modules
from langchain_openai import OpenAIEmbeddings  # Import OpenAI embeddings from LangChain
from dotenv import load_dotenv  # To load environment variables from a .env file

# Load environment variables (e.g., OpenAI API key) from a .env file
load_dotenv()

# Initialize the OpenAI embedding model
# model='text-embedding-3-large' specifies the version of the embedding model
# dimensions=32 reduces the embedding size to 32-dimensions (customizable based on use-case)
embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

# Embed a sample query/text into a 32-dimensional vector representation
result = embedding.embed_query("Delhi is the capital of India")

# Print the resulting embedding vector
print(str(result))
