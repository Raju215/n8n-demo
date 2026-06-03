import streamlit as st
from dotenv import load_dotenv

# LLM
from langchain_groq import ChatGroq

#### Tools
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun

# Agent
from langchain.agents import initialize_agent, AgentType

# Callback
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# ---------------------------------------------------------------------------
# Load Environment
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Tool Setup
# ---------------------------------------------------------------------------
# FIX 1: Increase doc_content_chars_max so the LLM gets enough context
#         to form a Final Answer instead of looping.
arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
)

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
)

tools = [arxiv, wiki]

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🔎 LangChain - Chat with Search")

st.markdown("""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts 
and actions of an agent in an interactive Streamlit app.  
Try more LangChain 🤝 Streamlit Agent examples at 
https://github.com/langchain-ai/streamlit-agent
""")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# ---------------------------------------------------------------------------
# Chat History
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------------------------------------------------------------------
# Chat Input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask anything..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.warning("Please enter your Groq API Key in the sidebar.")
        st.stop()

    # LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        streaming=True,
    )

    # FIX 2: Add max_iterations and early_stopping_method to prevent infinite loops.
    # FIX 3: Add a system prompt nudging the agent to conclude after 1-2 tool calls.
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        max_iterations=5,                        # hard stop after 5 steps
        early_stopping_method="generate",        # ask LLM to wrap up if limit hit
        agent_kwargs={
            "prefix": (
                "You are a helpful assistant. Use tools to look up information, "
                "then provide a concise Final Answer. "
                "Do NOT call the same tool twice with the same input."
            )
        },
    )

    # -----------------------------------------------------------------------
    # Run Agent
    # -----------------------------------------------------------------------
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        try:
            response = search_agent.run(prompt, callbacks=[st_cb])
        except Exception as e:
            response = f"⚠️ Error: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
