# Multilingual PDF Study Copilot

A web app that summarizes and answers questions about PDF documents in multiple languages. It can condense complex texts such as lecture notes, textbooks, or research papers into concise summaries, providing clear insights in the original language of the document.

**As it runs locally, your data is 100% safe**

try the app here: [Study Copilot](https://huggingface.co/spaces/javiialmendras/StudyCopilot)

![App](https://raw.githubusercontent.com/JavieraAlmendrasVilla/Files-summarizer/main/Study%20Copilot.jpg)

---

## Features

- Upload PDFs in **any language** and get summaries or answers in the same language.
- Ask questions about the PDF content and receive accurate responses.
- Summarizes complex topics into clear, digestible text.
- Powered by LangChain, Ollama LLM, and HuggingFace embeddings.
- Multilingual support for global usage.

---

## How It Works

1. **PDF Processing:** Splits the uploaded PDF into manageable text chunks.  
2. **Embedding Generation:** Converts each chunk into vector embeddings using HuggingFace.  
3. **Retrieval:** Chroma vector store retrieves the most relevant chunks for your query.  
4. **LLM Response:** Ollama LLM generates concise answers or summaries based solely on the PDF content.

---

## Requirements

- Python 3.10+
- Virtual environment recommended

Install dependencies:

```bash
pip install -r requirements.txt
````

---

## Usage

1. Clone or download the project.
2. Activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
python app.py
```

5. Open the Gradio interface in your browser:

   * Upload a PDF.
   * Enter a question about the content.
   * Receive summaries or answers instantly in the PDF's language.

---

## Example

* Upload a PDF in French about statistical methods.
* Ask: *"Quels sont les principaux coefficients de régression?"*
* Receive a concise answer or summary in French.

---

## Notes

* Answers and summaries are strictly based on the uploaded content; the tool does **not generate information outside the PDF**.
* Supports **any language** recognized by the underlying LLM.

---

## Dependencies

* [Gradio](https://gradio.app/) – Web interface
* [LangChain](https://www.langchain.com/) – LLM orchestration
* [LangChain Ollama](https://github.com/langchain-ai/llama) – LLM backend
* [HuggingFace Embeddings](https://huggingface.co/) – Embedding generation
* [Chroma](https://www.trychroma.com/) – Vector store for retrieval
* [pypdf](https://pypdf.readthedocs.io/) – PDF parsing




