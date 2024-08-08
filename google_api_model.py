# step 1.  Creating the Chat Model and configure the api key

from langchain_google_genai import ChatGoogleGenerativeAI

api_key = ""

with open('key.txt', 'r') as file:
    api_key = file.read()

model = ChatGoogleGenerativeAI(
      model = "gemini-1.5-flash", 
      google_api_key = api_key)

# step 2. Creating the ChatTemplate for user input

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=" You are a Chatbot designed for human conversation"),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}")
    ]
)

# step 3. Initializing the memory

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key = "chat_history", return_messages = True)

# Step 4. Creating an Output parser

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()


# RunnablePassthrough: RunnablePassthrough on its own allows you to pass inputs unchanged.

from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def get_messages_from_history(human_input):
  return memory.load_memory_variables(human_input)['chat_history']

chain = RunnablePassthrough.assign(
    chat_history = RunnableLambda(get_messages_from_history)
) | chat_prompt_template | model | output_parser

# Step 6. Invoke chat_history with human_input

query = {"human_input" : "what is your name?"}

response = chain.invoke(query)

print(response)

# Step 7. Store the results

memory.save_context({"human_input":query["human_input"]}, {"outputs":response})

# Step 8. running steps 6 and 7 in a loop

while True:
    
    query = {"human_input" : input('Enter your Query : ')}
    print(f"*User : {query['human_input']}")
    
    if query["human_input"].lower() in ["quit", "bye", "goodbye", "stop"]:
        break
        
    response = chain.invoke(query)
    print(f"*AI : {response}")
    
    memory.save_context(query, {"outputs":response})