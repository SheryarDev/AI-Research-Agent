
[Goal Description]
Build a persistent AI Research Assistant that tracks user research, remembers past conversations, and suggests related scientific papers. The system uses Groq for high-speed LLM inference and LangChain for orchestration and memory management.

Proposed Changes
Core Architecture
The system will be composed of three main layers:

Memory Layer: Uses a vector database (ChromaDB) to store "Semantic Memory" (embeddings of past conversations and research notes).
Research Layer: Integration with Arxiv/Semantic Scholar APIs to fetch papers and manage a "Reading List".
Inference Layer: LangChain's ChatGroq model with a custom prompt designed for research synthesis.
[Component Name]
[NEW] 
manager.py
Handles interaction with the vector database to store and retrieve past conversation contexts.

store_context(): Embeds and saves session data.
fetch_relevant_history(): Performs semantic search to find previous relevant research sessions.
[NEW] 
researcher.py
Manages paper discovery and reading status.

search_papers(): Queries Arxiv API.
get_related_work(): Suggests papers based on the current context and "Memory Layer".
[NEW] 
app.py
The entry point of the application (Streamlit-based) for a smooth UI.

Chat interface for interacting with the assistant.
Sidebar for viewing "Papers Read" and "Recent Sessions".
Verification Plan
Automated Tests
Test semantic search retrieval: Verify that a query about a past session returns relevant fragments.
Test Arxiv integration: Verify that paper metadata is correctly parsed and stored.
Manual Verification
Start a research session on "Quantum Computing".
End the session.
Start a new session on "AI for Physics" and check if the assistant suggests Quantum Computing papers or recalls previous discussions.
