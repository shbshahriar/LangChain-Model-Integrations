from langchain_openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (like your OpenAI API key)
load_dotenv()

# Initialize the OpenAI model (completion-style)
llm = OpenAI(
    model='gpt-3.5-turbo-instruct',  # ✅ Completion-based model (good for direct answers, not chat)
#    temperature=0.5,                 # 🎯 Controls randomness: 0 = more factual, 1 = more creative
#    max_tokens=100,                  #✂️ Limit the output length (prevents long, costly responses)
#    timeout=20,                      # ⏱ Optional: fail fast if OpenAI doesn't respond in 20 seconds
#    max_retries=2                    # 🔁 Retry a couple of times on errors (like rate limits)
)

# Run a prompt through the model
result = llm.invoke("What is the capital of Bangladesh?")

# Print the model's response
print(result)
