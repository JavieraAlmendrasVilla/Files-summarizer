import gradio as gr
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------------- LLM & Prompt Setup ----------------
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.1,
    top_k=10,
    top_p=0.1,
)

# Prompt for answering questions
qa_prompt_template = """Use the information from the document to answer questions about the file. 
If you don't know the answer, just say that you don't know, do not make up an answer.

{context}

Question: {question}
"""
QA_PROMPT = PromptTemplate(
    template=qa_prompt_template,
    input_variables=["context", "question"]
)

# Prompt for summarizing the PDF
summary_prompt_template = """Summarize the following document into a concise 1–2 page study guide.
Include:
- Key concepts
- Important formulas
- Tables or data summaries
- Examples if available

{context}
"""
SUMMARY_PROMPT = PromptTemplate(
    template=summary_prompt_template,
    input_variables=["context"]
)


# ---------------- Function to load PDF and setup retriever ----------------
def setup_pdf(file):
    loader = PyPDFLoader(file.name)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)

    # Create a retrieval chain for questions
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=docsearch.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=False
    )

    # Create a retrieval chain for summaries
    summary_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=docsearch.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": SUMMARY_PROMPT},
        return_source_documents=False
    )

    return qa_chain, summary_chain


# ---------------- Functions to handle queries and summaries ----------------
def generate_summary(file):
    _, summary_chain = setup_pdf(file)
    return summary_chain.run("Please summarize the document.")


def answer_query(file, question):
    qa_chain, _ = setup_pdf(file)
    return qa_chain.run(question)


# ---------------- Gradio Interface ----------------
with gr.Blocks() as iface:
    gr.Markdown("# PDF Study Copilot")
    gr.Markdown("Upload a PDF to either generate a 1–2 page study guide or ask questions about its content.")

    with gr.Tab("Generate Summary"):
        pdf_input_summary = gr.File(label="Upload PDF")
        summary_output = gr.Textbox(label="Summary", lines=20)
        summary_button = gr.Button("Generate Summary")
        summary_button.click(fn=generate_summary, inputs=pdf_input_summary, outputs=summary_output)

    with gr.Tab("Ask Questions"):
        pdf_input_qa = gr.File(label="Upload PDF")
        question_input = gr.Textbox(label="Ask a question about the PDF")
        answer_output = gr.Textbox(label="Answer")
        ask_button = gr.Button("Get Answer")
        ask_button.click(fn=answer_query, inputs=[pdf_input_qa, question_input], outputs=answer_output)

iface.launch()
