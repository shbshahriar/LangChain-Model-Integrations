# LangChain-Model-Integrations

This project demonstrates the use of various LangChain integrations with popular AI models and APIs for tasks like embeddings, document similarity, and chat-based interactions. It includes examples for OpenAI, Hugging Face, Anthropic, and Google Generative AI.

## Project Structure

The project is organized into the following directories:

### 1. `1_LLMS`
Contains examples of using OpenAI's language models for basic text generation tasks.
- **`1_llm_demo.py`**: Demonstrates the use of OpenAI's GPT-3.5-turbo-instruct model for generating responses to prompts.

### 2. `2_Chat_models`
Contains examples of chat-based interactions using various models.
- **`1_chat_model_openai.py`**: Uses OpenAI's GPT-4 for creative chat-based tasks.
- **`2_chat_model_anthropic.py`**: Demonstrates the use of Anthropic's Claude model for chat interactions.
- **`3_chat_model_google.py`**: Integrates Google's Gemini model for chat-based tasks.
- **`4_chat_model_hf_api.py`**: Uses Hugging Face's API for chat-based tasks.
- **`5_chat_model_hf_local.py`**: Demonstrates running Hugging Face models locally using LangChain.

### 3. `3_Embedded_models`
Contains examples of embedding generation and document similarity tasks.
- **`1_embedding_openai_query.py`**: Generates embeddings for a query using OpenAI's embedding model.
- **`2_embedding_openai_docs.py`**: Generates embeddings for documents using OpenAI's embedding model.
- **`3_embedding_hf_local.py`**: Demonstrates embedding generation using Hugging Face models locally.
- **`4_document_similarity.py`**: Computes document similarity using OpenAI embeddings.
- **`5_document_similarity_hf_local.py`**: Computes document similarity using Hugging Face embeddings locally.

### Other Files
- **`HF_login.py`**: Logs into Hugging Face using an API token.
- **`.env`**: Stores API keys for OpenAI, Anthropic, Google, and Hugging Face.
- **`requirements.txt`**: Lists the Python dependencies required for the project.
- **`.vscode/launch.json`**: Contains VS Code launch configurations for debugging.

## File Structure

The project is organized as follows:

```
Langchain_models/
├── 1_LLMS/
│   └── 1_llm_demo.py
├── 2_Chat_models/
│   ├── 1_chat_model_openai.py
│   ├── 2_chat_model_anthropic.py
│   ├── 3_chat_model_google.py
│   ├── 4_chat_model_hf_api.py
│   └── 5_chat_model_hf_local.py
├── 3_Embedded_models/
│   ├── 1_embedding_openai_query.py
│   ├── 2_embedding_openai_docs.py
│   ├── 3_embedding_hf_local.py
│   ├── 4_document_similarity.py
│   └── 5_document_similarity_hf_local.py
├── .vscode/
│   └── launch.json
├── HF_login.py
├── requirements.txt
├── .env
└── README.md
```

## Setup Instructions

1. Clone the repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory (if not already present) and add your API keys:
   ```env
   OPENAI_API_KEY = "your-openai-api-key"
   ANTHROPIC_API_KEY = "your-anthropic-api-key"
   GOOGLE_API_KEY = "your-google-api-key"
   HUGGINGFACE_API_KEY = "your-huggingface-api-key"
   ```
4. Run the desired script using Python:
   ```bash
   python path/to/script.py
   ```

## Usage Examples

### Running a Chat Model
To run the OpenAI chat model:
```bash
python 2_Chat_models/1_chat_model_openai.py
```

### Generating Document Embeddings
To generate embeddings using Hugging Face locally:
```bash
python 3_Embedded_models/3_embedding_hf_local.py
```

### Computing Document Similarity
To compute similarity between a query and documents using OpenAI embeddings:
```bash
python 3_Embedded_models/4_document_similarity.py
```

## Notes
- Ensure that the Hugging Face cache directory (`HF_HOME`) is set correctly in the scripts if you are using local models.
- Replace placeholder API keys in the `.env` file with your actual keys.

## License
This project is for educational purposes. Ensure compliance with the terms of use for the respective APIs and models.

## Acknowledgments
- [LangChain](https://github.com/hwchase17/langchain)
- [OpenAI](https://openai.com/)
- [Hugging Face](https://huggingface.co/)
- [Anthropic](https://www.anthropic.com/)
- [Google Generative AI](https://cloud.google.com/generative-ai)
