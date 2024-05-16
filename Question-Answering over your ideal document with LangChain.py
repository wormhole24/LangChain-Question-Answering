#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install --upgrade langchain openai -q')
get_ipython().system('pip install unstructured -q')
get_ipython().system('pip install unstructured[local-inference] -q')
get_ipython().system('pip install detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2 -q')
get_ipython().system('apt-get install poppler-utils')
get_ipython().system('pip install tiktoken -q')
get_ipython().system('pip install chromadb -q')
get_ipython().system('pip install pinecone-client -q')


# In[ ]:


import os
os.environ["OPENAI_API_KEY"] = " YOUR API KEY HERE "


# Let's start with question and answering over a single document here( e.g. 'txt' file), in case there is no memory for the model, thus it would not refer to your earlier questions :

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


location = "PATH TO YOUR FILE"
address = os.path.join(location)


# Loading the text file here, and tokenizing them by the 'TextLoader()' function
# Here the llm is ”text-davinci-3″ by default

# In[ ]:


from langchain.document_loaders import TextLoader
loader = TextLoader(address)


# Creating index out of the text

# In[ ]:


from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator().from_loaders([loader])


# Asking your questions

# In[ ]:


query = "YOUR QUESTION HERE"
index.query(query)


# Now we try to add Question and Answering considering memory, also we elevate the llm model to "gpt-3.5-turbo" 

# In[ ]:


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import  load_summarize_chain


# In[ ]:


directory = " Path to your file "
Booklet = os.path.join(directory)
loader = TextLoader(Booklet)
docy = loader.load()


# In[ ]:


llm = OpenAI(openai_api_key=" Your API key here ", model_name= " gpt-3.5-turbo " )
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter

#splitting the text file to avoid  exceeding the token limit
char_text_splitter = RecursiveCharacterTextSplitter(chunk_size =500, chunk_overlap = 0)

doc = char_text_splitter.split_documents(docy)
print(len(doc))


embeddings = OpenAIEmbeddings()
#chroma vector database
vectorstore = Chroma.from_documents(doc, embeddings)


# In[ ]:


from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# In[ ]:


qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory)


# In[ ]:


query = "Your question"
result = qa({"question": query})
result["answer"]

