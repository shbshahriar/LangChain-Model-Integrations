# Import the Gemini chat model wrapper from LangChain for Google Generative AI
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from a .env file (like your GOOGLE_API_KEY)
from dotenv import load_dotenv
load_dotenv()

# Initialize the Gemini 1.5 Pro model
model = ChatGoogleGenerativeAI(
    model='gemini-1.5-pro',       # ✅ Google's latest and most powerful model as of 2024

    # Optional parameters you can use later by uncommenting:
    # temperature=0.7,            # 🎯 Controls creativity: 0 = deterministic, 1 = highly creative
    # max_tokens=300,             # ✂️ Limits the length of the generated output (controls cost & focus)
    # top_p=1.0,                  # 📊 Nucleus sampling: considers tokens with cumulative probability top_p
    # top_k=40,                   # 🔢 Considers the top_k most likely tokens (used for fine-tuning output diversity)
    # stop=["\n"],                # 🛑 Optionally define stop sequences to control where output ends
    # timeout=20,                 # ⏱ Set max wait time for API response (helps avoid hangs)
    # max_retries=2,              # 🔁 Retry the request on failure (e.g., network issues, rate limits)
    # api_key="your-key-here",    # 🔐 Hardcode your API key (not recommended — use env var instead)
)

# Send a prompt to the model and store the response
result = model.invoke('write a poem about the ocean')


# Print only the generated content from the response
print(result.content)

