from flask import Flask, request, render_template, jsonify
import openai
from data.dataprovider import key, hg_key
from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain, ConversationalRetrievalChain, StuffDocumentsChain
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationChain
import re
from langchain.chains.conversation.memory import ConversationBufferMemory


app = Flask(__name__)

memory = ConversationBufferMemory() 

llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    },
    huggingfacehub_api_token= hg_key,# Replace with your actual huggingface token
)

from langchain_core.prompts import HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, StringPromptTemplate

# Define your rag chatbot function
def chat_with_rag(message):    
    
    conversation_sum = ConversationChain(llm=llm, memory=memory,verbose=True)
    result = conversation_sum(message)    
      
    return result

# Define your Flask routes
@app.route('/')
def home():
    return render_template('bot_1.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_input']
    bot_message = chat_with_rag(user_message)
    
    pattern = r"AI: ([\s\S]+?)(?=Human:|$)"
    matches = re.findall(pattern, bot_message['response'])

    if matches:
        last_ai_message = matches[-1].strip()        
        return jsonify({'response': last_ai_message})
    else:
        
        return jsonify({'response': "No AI messages found."})
    

if __name__ == '__main__':
    app.run(debug=True)
