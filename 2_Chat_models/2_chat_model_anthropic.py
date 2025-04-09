# Import Claude model integration from LangChain's Anthropic wrapper
from langchain_anthropic import ChatAnthropic

# Load environment variables (like ANTHROPIC_API_KEY) from .env file
from dotenv import load_dotenv
load_dotenv()

# Initialize Claude 3.5 Sonnet model from Anthropic
model = ChatAnthropic(
    model='claude-3-5-sonnet-20241022',  # ✅ Claude 3.5 Sonnet — fast & smart, released Oct 2024

    # Optional—but useful—parameters you might want later:
    # temperature=0.7,        # 🎯 Creativity control (0 = deterministic, 1 = more random)
    # max_tokens=300,         # ✂️ Limit response length (saves cost + avoids overly long replies)
    # timeout=20,             # ⏱ How long to wait for API response before erroring out
    # max_retries=2,          # 🔁 Number of retries on failure (e.g., timeout or API rate limit)
    # api_key="your-key",     # 🔐 If you want to hardcode your Anthropic key instead of using env vars
    # base_url="your-url",    # 🌐 For custom Claude-compatible endpoints (e.g., local proxy, fine-tuned)
)

# Send a prompt to the model and store the response
result = model.invoke('What is the capital of India')

# Print the result content (Claude uses `.content` for response)
print(result.content)
