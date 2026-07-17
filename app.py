import os
import streamlit as st
import pymupdf4llm
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

# Set up page config
st.set_page_config(page_title="Med AI Clinical Advisor", page_icon="💊", layout="wide")
st.title("💊 MEDMind AI Clinical Advisor")
st.caption("AI Agent with Hybrid Parent-Document Retrieval over the drug Monograph")

# --- STEP 1: CACHED RETRIEVER SETUP ---
@st.cache_resource(show_spinner="Initializing Clinical Retriever (this may take a minute)...")
def initialize_agent_and_retriever(pdf_path: str):
    page_data = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)

    headers_to_split_on = [
        ("##", "Section"),
        ("###", "Subsection"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    parent_docs = []
    for page in page_data:
        page_num = page["metadata"].get("page_number", "unknown")
        page_splits = markdown_splitter.split_text(page["text"])
        for doc in page_splits:
            doc.metadata["page"] = page_num
            parent_docs.append(doc)

    for i, doc in enumerate(parent_docs):
        doc.metadata["parent_id"] = f"parent_{i}"

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    vectorstore = Chroma(collection_name="metformin_child_chunks", embedding_function=embeddings)
    store = InMemoryStore()

    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        search_kwargs={"k": 3},
        key_to_id="parent_id",
    )
    parent_retriever.add_documents(parent_docs)

    bm25_retriever = BM25Retriever.from_documents(parent_docs)
    bm25_retriever.k = 3

    hybrid_retriever = EnsembleRetriever(
        retrievers=[parent_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    return hybrid_retriever

# --- STEP 2: LOAD ASSETS & DEFINE AGENT ---
pdf_file = "metformin.pdf"

# Golden Dataset for evaluation
GOLDEN_DATASET = [
    {
        "Question": "What is the action for an eGFR of 35?",
        "Expected Key Information": "Initiation is not recommended. For patients already taking metformin, consider a 50% dose reduction and monitor eGFR every 3 months.",
        "Section": "6. Warnings, Precautions & Drug Interactions"
    },
    {
        "Question": "What is the maximum recommended dose for pediatric patients?",
        "Expected Key Information": "2,000 mg/day in divided doses. Extended-release forms are not approved.",
        "Section": "4. Dosage and Administration"
    },
    {
        "Question": "Is metformin contraindicated for an eGFR below 30?",
        "Expected Key Information": "Yes, it is contraindicated. Discontinue immediately.",
        "Section": "5. Contraindications / 6. Renal Impairment Guidelines"
    },
    {
        "Question": "What are the common gastrointestinal adverse reactions and their percentages?",
        "Expected Key Information": "Diarrhea (53%), Nausea/Vomiting (26%), Flatulence (12%), Abdominal discomfort (6%).",
        "Section": "7. Adverse Reactions"
    },
    {
        "Question": "What is the recommended timing for monitoring Vitamin B12 levels?",
        "Expected Key Information": "Periodic monitoring is advised every 2 to 3 years.",
        "Section": "7. Adverse Reactions"
    }
]

try:
    hybrid_retriever = initialize_agent_and_retriever(pdf_file)

    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("🔑 Groq API Key not found! Please set it as an environment variable or in `.streamlit/secrets.toml`.")
        st.stop()

    llm = ChatGroq(
        model='llama-3.3-70b-versatile',
        temperature=0.0,
        api_key=groq_api_key
    )

    @tool
    def metformin_tool(question: str) -> str:
        """Searches the Metformin clinical monograph and guidelines."""
        relevant_chunks = hybrid_retriever.invoke(question)
        return "\n\n".join(
            f"[Section: {c.metadata.get('Section', 'N/A')} | Page: {c.metadata.get('page', 'N/A')}]\n{c.page_content}"
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
    agent = create_react_agent(llm, tools, prompt=system_message)

except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()


# --- STEP 3: STREAMLIT UI WITH EVALUATION TAB ---
tab1, tab2 = st.tabs(["💬 Chat Assistant", "📊 Agent Evaluation"])

# --- TAB 1: CHAT INTERFACE ---
with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your MedAI clinical assistant. Ask me anything about metformin guidelines, dosing, or contraindications."}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask a clinical question (e.g., 'What is the action for an eGFR of 35?')"):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Consulting guidelines..."):
                try:
                    formatted_history = []
                    for msg in st.session_state.messages:
                        if msg["role"] == "user":
                            formatted_history.append(("user", msg["content"]))
                        elif msg["role"] == "assistant":
                            formatted_history.append(("assistant", msg["content"]))

                    response = agent.invoke({"messages": formatted_history})
                    final_reply = response['messages'][-1].content

                    st.markdown(final_reply)
                    st.session_state.messages.append({"role": "assistant", "content": final_reply})

                except Exception as e:
                    st.error(f"An error occurred: {e}")

# --- TAB 2: GOLDEN DATASET EVALUATION ---
with tab2:
    st.header("Monograph Golden Data Benchmark")
    st.write("Run the automated benchmark suite against the clinical ground truth records.")

    if st.button("🚀 Run System Evaluation"):
        eval_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, item in enumerate(GOLDEN_DATASET):
            status_text.text(f"Evaluating Question {idx + 1}/{len(GOLDEN_DATASET)}...")
            try:
                # Query the system with zero chat history to test factual grounding purely
                res = agent.invoke({"messages": [("user", item["Question"])]})
                agent_output = res['messages'][-1].content
            except Exception as e:
                agent_output = f"ERROR: {e}"
            
            eval_results.append({
                "Clinical Question": item["Question"],
                "Target Ground Truth (Expected)": item["Expected Key Information"],
                "Agent Response": agent_output,
                "Monograph Section": item["Section"]
            })
            progress_bar.progress((idx + 1) / len(GOLDEN_DATASET))
            
        status_text.text("✅ Evaluation completed!")
        
        # Convert to Pandas DataFrame for presentation
        df_results = pd.DataFrame(eval_results)
        
        # Display nicely in a table format
        st.subheader("Benchmark Results Matrix")
        st.dataframe(
            df_results, 
            use_container_width=True,
            column_config={
                "Agent Response": st.column_config.TextColumn(width="large"),
                "Target Ground Truth (Expected)": st.column_config.TextColumn(width="medium")
            }
        )
