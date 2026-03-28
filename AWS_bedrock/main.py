from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Bedrock
import boto3
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

aws_access_key_id = (os.getenv("AWS_ACCESS_KEY_ID") or "").strip().strip("'\"")
aws_secret_access_key = (os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip().strip("'\"")
region_name = (os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION_NAME") or "us-east-1").strip().strip("'\"")
if region_name.lower() == "global":
    region_name = "us-east-1"

# bedrock client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)
model_id = (os.getenv("BEDROCK_MODEL_ID") or "anthropic.claude-3-sonnet-20240229-v1:0").strip().strip("'\"")
llm = Bedrock(
    client=bedrock,
    model_id=model_id,
    model_kwargs={"temperature": 0.7, "max_tokens": 300}
)
     
def my_chatbot(language, user_text):
    prompt = PromptTemplate(
        input_variables=["language", "user_text"],
        template="You are a helpful assistant that can speak in {language}. Answer the following question: {user_text}"
    )
    formatted_prompt = prompt.format(language=language, user_text=user_text)
    response = llm.invoke(formatted_prompt)
    
    return response              

# UI in streamlit
st.title("Multilingual Chatbot with AWS Bedrock")
language = st.selectbox("Select a language", ["English", "Nepali", "French", "German", "Chinese"])
if language:
    user_text = st.sidebar.text_area(label="what is your question?",
                                   max_chars=500)
    
    if user_text:
        try:
            response = my_chatbot(language, user_text)
            st.write(response)
        except Exception as exc:
            st.error(
                "Bedrock request failed. Check AWS region/model access and credentials. "
                f"Details: {exc}"
            )
      
    