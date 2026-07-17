import os
import streamlit as st
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

# Set up page config
st.set_page_config(page_title="Med AI Clinical Advisor", page_icon="💊", layout="wide")
st.title("💊 Med AI Clinical Advisor")
st.caption("AI Agent with Hybrid Parent-Document Retrieval over the drug Monograph")

# --- STEP 1: CACHED RETRIEVER SETUP ---
@st.cache_resource(show_spinner="Initializing Clinical Retriever (this may take a minute)...")
def initialize_agent_and_retriever(pdf_path: str):
    # 1. Read and parse PDF
   
    pdf_markdown_content = pymupdf4llm.to_markdown(pdf_path)

    # 2. Structure-Aware Parsing (Markdown Header Splitting)
    headers_to_split_on = [
        ("##", "Section"),
        ("###", "Subsection"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    parent_docs = markdown_splitter.split_text(pdf_markdown_content)

    # Add parent identifiers
    for i, doc in enumerate(parent_docs):
        doc.metadata["parent_id"] = f"parent_{i}"

    # 3. Embedding and Retriever Components
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    vectorstore = Chroma(collection_name="metformin_child_chunks", embedding_function=embeddings)
    store = InMemoryStore()

    # Parent Retriever
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        search_kwargs={"k": 3},
        key_to_id="parent_id",
    )
    parent_retriever.add_documents(parent_docs)

    # BM25 Sparse Retriever
    bm25_retriever = BM25Retriever.from_documents(parent_docs)
    bm25_retriever.k = 3

    # Hybrid Ensemble
    hybrid_retriever = EnsembleRetriever(
        retrievers=[parent_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    return hybrid_retriever

# --- STEP 2: LOAD ASSETS & DEFINE AGENT ---
pdf_file = "metformin.pdf"

try:
    # Warm up / fetch our retriever
    hybrid_retriever = initialize_agent_and_retriever(pdf_file)

    # Set up LLM with safety check for API keys
    groq_api_key = st.secrets.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("🔑 Groq API Key not found! Please set it as an environment variable or in `.streamlit/secrets.toml`.")
        st.stop()

    llm = ChatGroq(
        model='llama-3.1-8b-instant',
        temperature=0.0,
        api_key=groq_api_key
    )

    # Declare the tool inside the loaded scope so it references the cached retriever
    @tool
    def metformin_tool(question: str) -> str:
        """Searches the Metformin clinical monograph and guidelines.
        Use this tool to find clinical pharmacology, warnings, dosage, lactic acidosis risk,
        eGFR adjustments, and contraindications. Input must be a clear search query."""
        relevant_chunks = hybrid_retriever.invoke(question)
        return "\n\n".join(
        f"[Section: {c.metadata.get('Section','?')} | Page: {c.metadata.get('page','?')}]\n{c.page_content}"
        for c in relevant_chunks
)

    tools = [metformin_tool]
    system_prompt = (
        "You are a clinical decision support assistant. Your answers must be completely grounded "
        "in the provided context. If the context does not contain the answer, explicitly state "
        "'I cannot find this information in the provided guidelines.' Do not assume or extrapolate. "
        "Always cite the source and page number in your response using markdown [Source, Page]."
    )
    system_message = SystemMessage(content=system_prompt)

    # Initialize Agent
    agent = create_react_agent(llm, tools, prompt=system_message)

except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()



# --- STEP 4: SESSION STATE & CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your MedAI clinical assistant. Ask me anything about metformin guidelines, dosing, or contraindications."}
    ]

# Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if user_input := st.chat_input("Ask a clinical question (e.g., 'What is the action for an eGFR of 35?')"):
    # Render user's message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Call agent and generate assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Consulting guidelines..."):
            try:
                # Format message history for LangGraph ReAct agent
                formatted_history = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        formatted_history.append(("user", msg["content"]))
                    elif msg["role"] == "assistant":
                        formatted_history.append(("assistant", msg["content"]))

                # Invoke the agent
                response = agent.invoke({"messages": formatted_history})

                # Extract the final answer content from the agent's graph execution
                final_reply = response['messages'][-1].content

                st.markdown(final_reply)
                st.session_state.messages.append({"role": "assistant", "content": final_reply})

            except Exception as e:
                st.error(f"An error occurred while processing your request: {e}")
