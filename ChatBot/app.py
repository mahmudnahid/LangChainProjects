## Chatbot_LLM
import streamlit as st

from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate

from dotenv import load_dotenv
load_dotenv()
import os

HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN')
os.environ["HF_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

## Streamlit UI
st.set_page_config(page_title="Conversational Chatbot")
st.header("Hey, Let's Chat")

if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [SystemMessage(content="You're a helpful AI assistant.")]

def get_prompt_template(): 
    template_messages = [
        # SystemMessage(content="You're a helpful AI assistant."),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
    prompt_template = ChatPromptTemplate.from_messages(template_messages)
    return prompt_template


def get_model():
    ## Chat Model
    chatllm = HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        task="text-generation",
        model_kwargs={
            "temperature": 0.6,
            "repetition_penalty": 1.03,
        },
    )

    return ChatHuggingFace(llm=chatllm)


## Function to load model and get respones
def get_chatmodel_response(query):
    st.session_state['flowmessages'].append(HumanMessage(content=query))

    model = get_model()
    chatprompt =  get_prompt_template() 
    chain = chatprompt|model
    response = chain.invoke({"query": query})
    print(response.content)

    st.session_state['flowmessages'].append(AIMessage(content=response.content))
    return response.content

input = st.text_input("Input: ",key="input")
response = get_chatmodel_response(input)

submit=st.button("Ask the question")

## If ask button is clicked
if submit:
    st.subheader("The Response is")
    st.write(response)