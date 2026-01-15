from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser  
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langserve import add_routes

load_dotenv()

# Check if variables are loaded 
groq_key = os.getenv('GROQ_API_KEY')
if not groq_key :
    raise ValueError("Missing required environment variables in .env file")
os.environ['GROQ_API_KEY'] = groq_key             
LLM=ChatGroq(model='llama-3.1-8b-instant',temperature=0.7, max_tokens=512)

generic_prompt="Translate following into {language}"
prompt=ChatPromptTemplate.from_messages(
    [("system",generic_prompt),
     ("user","{text}")]
    )

parser=StrOutputParser()

chain=prompt | LLM | parser

#app defination
app=FastAPI(title="Simple GenAI App as API",version="1.0",description="A simple Generative AI application exposed as an API using FastAPI and LangServe")

add_routes(app,chain, path="/chain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)