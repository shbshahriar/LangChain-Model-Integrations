# Import necessary classes for using HuggingFace models via LangChain
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables (e.g., HUGGINGFACEHUB_API_TOKEN)
from dotenv import load_dotenv
load_dotenv()

# Set up the base LLM from Hugging Face using a specific model repo
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # ✅ Lightweight chat model (open-source, 1.1B params)
    task="text-generation",                        # 📌 The task to perform: basic text generation

    # Optional parameters for more control:
    # temperature=0.7,                              # 🎨 Controls randomness (0 = deterministic, 1 = more creative)
    # max_new_tokens=100,                           # ✂️ Limits how long the response can be
    # top_k=50,                                     # 🎯 Limits next-token choices to top_k most likely
    # top_p=0.9,                                    # 📊 Nucleus sampling — more flexible than top_k
    # repetition_penalty=1.1,                       # 🔁 Discourages repeating the same phrases
    # stop_sequences=["\n"],                        # 🛑 Optional: define where model should stop generating
    # huggingfacehub_api_token="hf_xxx",            # 🔐 If not using env var, you can set your token directly here
)

# Wrap the raw endpoint in a chat-style interface (lets you use `.invoke()` like ChatGPT)
model = ChatHuggingFace(llm=llm)

# Invoke the model with a simple prompt
result = model.invoke("What is the capital of India")

# Print the result content (ChatHuggingFace returns an object with `.content`)
print(result.content)
