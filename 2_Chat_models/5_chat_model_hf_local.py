# Import classes to run Hugging Face models via LangChain (using local or cached setup)
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

# Set Hugging Face cache directory (downloads and stores models here)
os.environ['HF_HOME'] = 'D:/HF/Tiny LLama'  # 🗃️ Path where model + tokenizer files will be cached

# Load the TinyLlama model using Hugging Face pipelines (local or via HF Hub)
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',  # ✅ Small, efficient open-source chat model
    task='text-generation',                         # 📌 Specifies the task (can also be 'text2text-generation')

    # Customize inference behavior
    pipeline_kwargs=dict(
        temperature=0.5,       # 🎯 Controls randomness: 0 = more focused, 1 = more creative
        max_new_tokens=100     # ✂️ Limit the response length
        # top_k=50,            # 🔢 (Optional) Consider top-k tokens only
        # top_p=0.95,          # 📊 (Optional) Use nucleus sampling
        # repetition_penalty=1.1,  # 🔁 (Optional) Penalize repeating words/phrases
    )
)

# Wrap the base pipeline with a chat-friendly interface (so you can use .invoke)
model = ChatHuggingFace(llm=llm)

# Send a prompt to the model
result = model.invoke("What is the capital of India")

# Print the model's response content
print(result.content)
