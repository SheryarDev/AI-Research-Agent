import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from src.memory_manager import MemoryManager
from src.research_engine import ResearchEngine

# Load environment variables
load_dotenv()

# Initialize components
if "memory" not in st.session_state:
    st.session_state.memory = MemoryManager()
if "researcher" not in st.session_state:
    st.session_state.researcher = ResearchEngine()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# LLM Setup
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

st.set_page_config(page_title="AI Research Assistant", layout="wide")

st.title("📚 AI Research Assistant with Memory")
st.markdown("Your research sessions are stored in **Long-Term Memory** for future recall.")

# Sidebar for memory and tools
with st.sidebar:
    st.header("Memory & Tools")
    if st.button("Recall Past Context"):
        query = st.text_input("What would you like to recall?", "")
        if query:
            hits = st.session_state.memory.fetch_relevant_history(query)
            st.write("Relevant past findings:")
            for hit in hits:
                st.info(hit.page_content)
    
    st.divider()
    st.subheader("Persistent Reading List")
    # This would ideally be pulled from a DB, using memory for now
    if st.button("Search Arxiv"):
        search_query = st.text_input("Search papers:", key="arxiv_search")
        if search_query:
            papers = st.session_state.researcher.search_papers(search_query)
            for p in papers:
                st.write(f"- [{p['title']}]({p['url']})")

# Chat Interface
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)

if prompt := st.chat_input("Ask about your research..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 1. Fetch relevant history from memory
    past_memory = st.session_state.memory.fetch_relevant_history(prompt, k=2)
    memory_context = "\n".join([h.page_content for h in past_memory])
    
    # 2. Prepare LLM prompt
    full_prompt = f"""
    You are a helpful Research Assistant with access to the user's past notes.
    
    Past Context:
    {memory_context}
    
    User Query: {prompt}
    """
    
    # 3. Get AI Response
    response = llm.invoke(full_prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response.content)
    
    # 4. Store session in memory
    st.session_state.memory.store_context(
        f"User asked: {prompt}\nAI responded: {response.content}",
        {"type": "chat_session"}
    )
    
    # 5. Update UI chat history
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=response.content))
