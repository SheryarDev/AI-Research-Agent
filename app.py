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
    
    st.subheader("Recall Past Context")
    memory_query = st.text_input("What would you like to recall?", key="mem_recall_input")
    if st.button("Search Memory"):
        if memory_query:
            with st.spinner("Searching memory..."):
                hits = st.session_state.memory.fetch_relevant_history(memory_query)
                if hits:
                    st.write("Relevant past findings:")
                    for hit in hits:
                        st.info(hit.page_content)
                else:
                    st.warning("No relevant memory found.")
        else:
            st.warning("Please enter a query.")
    
    st.divider()
    
    st.subheader("Arxiv Research")
    search_query = st.text_input("Search papers:", key="arxiv_search_input")
    if st.button("Find Papers"):
        if search_query:
            with st.spinner("Searching Arxiv..."):
                try:
                    papers = st.session_state.researcher.search_papers(search_query)
                    if papers:
                        for p in papers:
                            st.markdown(f"**[{p['title']}]({p['url']})**")
                            st.caption(f"Authors: {', '.join(p['authors'])}")
                            with st.expander("Abstract"):
                                st.write(p['summary'])
                    else:
                        st.warning("No papers found.")
                except Exception as e:
                    st.error(f"Error searching Arxiv: {e}")
        else:
            st.warning("Please enter a search term.")

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
