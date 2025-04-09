# Import the ChatOpenAI class from langchain_openai to use OpenAI models in LangChain
from langchain_openai import ChatOpenAI

# Load environment variables (like your OpenAI API key) from a .env file
from dotenv import load_dotenv
load_dotenv()  # This ensures your OpenAI API key is securely loaded

# Initialize the GPT-4 model from OpenAI with creative settings
# - temperature=1.5 means the responses will be highly creative and less predictable
# - max_tokens=50 limits the length of the response to avoid overly long generations
model = ChatOpenAI(
    model='gpt-4',
    temperature=1.5,
    max_tokens=50
)

# Send a prompt to the model and store the result
# In this case, we're asking it to write a short 5-line poem about cricket
result = model.invoke("Write a 5 line poem on cricket")

# Print the generated poem to the console
print(result.content)
