# GenAI Learning Archive

Curated notebook collection for learning practical GenAI workflows: text preprocessing, classical NLP, Hugging Face pipelines, LangChain chains/agents, local Llama usage, and vector databases.

## Repository Structure

```text
.
├── AWS_bedrock/
│   └── requirements.txt
├── Data_handling/
│   ├── text_classification.ipynb
│   ├── text_preprocessing.ipynb
│   └── word_embeddings.ipynb
├── Falcon/
│   ├── FalconP_1_3b_.ipynb
│   └── Multi_doc_Retrieval.ipynb
├── HuggingFace/
│   ├── text_summarizer_with_huggingface.ipynb
│   └── text_to_speech_with_huggingface.ipynb
├── Langchian/
│   ├── Langchain.ipynb
│   └── RAG-/
│       └── notebook.ipynb
├── LLamaindex/
│   ├── Llamaindex.ipynb
│   └── Mistral.ipynb
├── Llama/
│   ├── Llama.ipynb
│   └── Llama2_with_langchain.ipynb
├── vectorDB/
│   ├── ChromaDB.ipynb
│   ├── Pinecone.ipynb
│   └── wavelate.ipynb
├── LICENSE
└── README.md
```

## Quick Start

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install baseline dependencies

```bash
pip install jupyterlab notebook ipykernel pandas numpy scikit-learn nltk gensim spacy transformers datasets accelerate
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4
```

### 3) Launch notebooks

```bash
jupyter lab
```

Open any `.ipynb` file from the file browser.

### 4) AWS Bedrock notebook dependencies (optional)

If you plan to work on Bedrock workflows, install the dedicated dependency list:

```bash
pip install -r AWS_bedrock/requirements.txt
```

## API Keys and Environment Variables

Several notebooks call hosted APIs. Set keys in your shell before launching Jupyter:

```bash
export OPENAI_API_KEY="your_openai_key"
export HUGGINGFACEHUB_API_TOKEN="your_hf_hub_token"
export HF_TOKEN="your_hf_token"                  # used by some HF/Colab flows
export PINECONE_API_KEY="your_pinecone_key"
export PINECONE_API_ENV="your_pinecone_env"      # example: gcp-starter
export WEAVIATE_API_KEY="your_weaviate_key"
```

Important:
- Do not hardcode secrets directly in notebook cells.
- Prefer environment variables and `os.environ.get(...)`.

## Tool-by-Tool Guide (Commands + APIs)

### AWS Bedrock (`AWS_bedrock/`)

Files:
- `AWS_bedrock/requirements.txt`

Install command used:

```bash
pip install -r AWS_bedrock/requirements.txt
```

Main libraries used:
- `boto3`
- `langchain`
- `langchain-community`
- `streamlit`
- `python-dotenv`

### Data Handling (`Data_handling/`)

Notebooks:
- `Data_handling/text_preprocessing.ipynb`
- `Data_handling/text_classification.ipynb`
- `Data_handling/word_embeddings.ipynb`

Typical install commands used:

```bash
pip install pandas numpy scikit-learn nltk gensim spacy
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4
```

Main APIs used:
- `nltk.download(...)`
- `nltk.tokenize.word_tokenize`, `nltk.tokenize.sent_tokenize`
- `nltk.corpus.stopwords.words`
- `nltk.stem.PorterStemmer`, `nltk.stem.WordNetLemmatizer`
- `spacy.load("en_core_web_sm")`
- `sklearn.feature_extraction.text.CountVectorizer`
- `sklearn.feature_extraction.text.TfidfVectorizer`
- `sklearn.model_selection.train_test_split`
- `sklearn.naive_bayes.GaussianNB`
- `sklearn.ensemble.RandomForestClassifier`
- `sklearn.metrics.accuracy_score`, `sklearn.metrics.confusion_matrix`
- `gensim.models.Word2Vec`

### Hugging Face (`HuggingFace/`)

Notebooks:
- `HuggingFace/text_summarizer_with_huggingface.ipynb`
- `HuggingFace/text_to_speech_with_huggingface.ipynb`

Typical install commands used:

```bash
pip install transformers datasets sentencepiece accelerate evaluate rouge_score
```

Main APIs used:
- `transformers.pipeline(...)`
- `transformers.AutoTokenizer.from_pretrained(...)`
- `transformers.AutoModelForSeq2SeqLM.from_pretrained(...)`
- `transformers.TrainingArguments`
- `transformers.Trainer`
- `transformers.DataCollatorForSeq2Seq`

Examples from notebooks:
- Summarization pipeline and Pegasus-style seq2seq fine-tuning workflow.
- Text-to-speech pipeline: `pipeline("text-to-speech", model="suno/bark-small", ...)`.

### LangChain (`Langchian/` and root integration)

Notebooks:
- `Langchian/Langchain.ipynb`
- `Llama2_with_langchain.ipynb`

Typical install commands used:

```bash
pip install langchain langchain-community langchain-experimental langchain-openai langchain-pinecone
pip install transformers accelerate bitsandbytes einops
```

Main APIs used:
- `langchain_experimental.agents.agent_toolkits.create_pandas_dataframe_agent(...)`
- `langchain.llms.HuggingFacePipeline`
- `langchain.chains.RetrievalQA.from_chain_type(...)`
- `transformers.pipeline("text2text-generation", ...)`

Notes:
- Some older cells use legacy imports (`from langchain.llms import ...`).
- If imports fail on latest LangChain, install matching legacy-compatible versions or update imports to current package paths.

### Llama (`Llama/`)

Notebook:
- `Llama/Llama.ipynb`

Typical install commands used:

```bash
pip install llama-cpp-python huggingface_hub
```

GPU build command used in notebook (Colab-style):

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.78 --force-reinstall
```

Main APIs used:
- `from llama_cpp import Llama`
- `Llama(model_path=..., n_ctx=..., ...)`
- Hugging Face model/token access via `huggingface_hub` and `HF_TOKEN`

### Vector Databases (`vectorDB/`)

Notebooks:
- `vectorDB/ChromaDB.ipynb`
- `vectorDB/Pinecone.ipynb`
- `vectorDB/wavelate.ipynb` (Weaviate-focused)

Typical install commands used:

```bash
pip install chromadb pinecone-client weaviate-client langchain langchain-community langchain-openai langchain-pinecone openai tiktoken pypdf
```

Main APIs used:
- Chroma:
	- `langchain.vectorstores.Chroma` / `langchain_community.vectorstores.Chroma`
	- `Chroma.from_documents(...)`
- Embeddings:
	- `langchain.embeddings.OpenAIEmbeddings`
	- `langchain_openai.OpenAIEmbeddings`
- Pinecone:
	- `pinecone.init(api_key=..., environment=...)`
- Retrieval and QA:
	- `langchain.chains.RetrievalQA.from_chain_type(...)`
	- `langchain.chains.question_answering.load_qa_chain(...)`
- Document processing:
	- `langchain.document_loaders.DirectoryLoader`, `TextLoader`
	- `langchain.text_splitter.RecursiveCharacterTextSplitter`

## Running a Specific Notebook

Example: open one notebook directly from CLI.

```bash
jupyter lab "HuggingFace/text_summarizer_with_huggingface.ipynb"
```

## Troubleshooting

- `ModuleNotFoundError`: install the missing package in the same environment as Jupyter kernel.
- `AuthenticationError` / `401`: verify API key env vars are set and valid.
- HF rate limits/download issues: set `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN`.
- LangChain import path errors: align code with installed LangChain version (legacy vs latest split packages).

## License

This repository is licensed under the terms in `LICENSE`.
