from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# Read the API Key
api_key = ""
with open('key.txt', 'r') as file:
    api_key = file.read().strip()

# Step 1: Creating the Chat Model and configuring the API key
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key
)

# Step 2: Creating the ChatTemplate for user input
chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage("You are Chatbot designed for human conversation..."),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}")
    ]
)

# Step 3: Initializing the memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Step 4: Creating an Output parser
output_parser = StrOutputParser()

# Step 5: RunnablePassthrough: RunnablePassthrough on its own allows you to pass inputs unchanged.
def get_messages_from_history(human_input):
    return memory.load_memory_variables(human_input)["chat_history"]

chain = RunnablePassthrough.assign(
    chat_history=RunnableLambda(get_messages_from_history)
) | chat_template | model | output_parser

# Streamlit setup
st.set_page_config(page_title="Chatbot")
st.title("AI Chatbot")

st.chat_message("assistant").write("Hi, How may I help you?")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# For taking user inputs
user_input = st.chat_input()

if user_input:
    query = {"human_input": user_input}
    response = chain.invoke(query)
    st.chat_message("user").write(user_input)
    st.chat_message("ai").write(response)
    
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.session_state["messages"].append({"role": "ai", "content": response})
    
    memory.save_context({"human_input": user_input}, {"outputs": response})