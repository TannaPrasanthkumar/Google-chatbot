from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

# Read the API Key
api_key = ""
with open('key.txt', 'r') as file:
    api_key = file.read().strip()

# Step 1: Creating the Chat Model and configuring the API key
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key
)

# Step 2: Creating the ChatTemplate for code review
chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage("You are an AI code reviewer. Your task is to review Python code provided by the user, identify any potential bugs or improvements, and suggest fixes."),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("Here is the Python code for review:\n\n{human_input}")
    ]
)

# Step 3: Initializing the memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Step 4: Creating an Output parser
output_parser = StrOutputParser()

# Step 5: RunnablePassthrough: To pass inputs unchanged
def get_messages_from_history(human_input):
    return memory.load_memory_variables(human_input)["chat_history"]

chain = RunnablePassthrough.assign(
    chat_history=RunnableLambda(get_messages_from_history)
) | chat_template | model | output_parser

# Streamlit setup
st.set_page_config(page_title="AI Code Reviewer", page_icon="💻", layout="centered")
st.title("💻 AI Code Reviewer")
st.markdown(
    """
    Submit your Python code for review and get feedback on potential bugs, errors, or improvements.
    The AI will also provide corrected code snippets for better clarity.
    """
)

# Sidebar with icons and GitHub link
with st.sidebar:
    st.title("🚀 AI Code Reviewer")
    st.markdown("### Made with ❤️ for Python enthusiasts!")
    st.markdown(
        """
        **Features:**
        - 🐍 Python code review
        - ✅ Bug detection
        - 💡 Fix suggestions
        """
    )
    add_vertical_space(2)
    st.markdown(
        """
        **GitHub Repository**:  
        [🌐 View Code on GitHub](https://github.com/TannaPrasanthkumar/AI-Code-Reviewer)
        """
    )
    add_vertical_space(2)
    st.markdown("📫 Contact: [tannaprasanthkumar76@gmail.com](mailto:tannaprasanthkumar76@gmail.com)")

# Display the conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("👤 User").write(msg["content"])
    elif msg["role"] == "ai":
        st.chat_message("🤖 AI").write(msg["content"])

# User input for Python code
st.markdown("### 📝 Enter your Python Code Below")
user_input = st.text_area("Write or paste your Python code here:", height=300, placeholder="Example: import numpy as np\nprint(np.random.randn(10))")

# Review button with functionality
if st.button("🔍 Review Code"):
    if user_input:
        query = {"human_input": user_input}
        response = chain.invoke(query)
        
        # Display user input and AI response
        st.chat_message("👤 User").write(f"Here is the Python code for review:\n\n{user_input}")
        st.chat_message("🤖 AI").write(response)
        
        # Append to conversation history
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["messages"].append({"role": "ai", "content": response})
        
        # Save context to memory
        memory.save_context({"human_input": user_input}, {"outputs": response})
    else:
        st.warning("⚠️ Please enter some Python code for review!")

# Footer for credits
st.markdown(
    """
    <hr>
    <div style='text-align: center;'>
        <p>Made with ❤️ by <strong>Tanna Prasanth kumar</strong> | © 2024</p>
    </div>
    """,
    unsafe_allow_html=True
)
