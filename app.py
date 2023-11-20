import gradio as gr
from langchain.document_loaders import PDFMinerLoader, PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import chromadb
import chromadb.config
from chromadb.config import Settings
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import uuid
from sentence_transformers import SentenceTransformer
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

# load the model
model_name = 'google/flan-t5-base'
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# to calculate text embeddings
ST_name = 'sentence-transformers/sentence-t5-base'
st_model = SentenceTransformer(ST_name)

# to store our embeddings and search
client = chromadb.Client()
collection = client.create_collection("my_db") 


def get_context(query_text):
    '''
    Given query in tokenized format, find its embeddings
    Search in Chroma DB
    and return results
    '''
    
    query_emb = st_model.encode(query_text)
    query_response = collection.query(query_embeddings=query_emb.tolist(), n_results=4)
    context = query_response['documents'][0][0]
    context = context.replace('\n', ' ').replace('  ', ' ')
    return context



def local_query(query, context):
    '''
    Given query (user response)
    Construct LLM query adding context to it
    Return response of LLM
    '''

    
    t5query = """Please answer the question based on the given context. 
    If you are not sure about your response, say I am not sure.
    Context: {}
    Question: {}
    """.format(context, query)

    # calculate embeddings for the query
    inputs = tokenizer(t5query, return_tensors="pt")

    outputs = model.generate(**inputs, max_new_tokens=20)
 
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)



   
   

def run_query(history, query):
    '''
    Run Gradio ChatInterface
    Given user response (query), find the most similar/related part to the question from the uploaded document
    Using Chroma search
    Update the query with context, and ask the question to LLM
    '''

    context = get_context(query) # find the related part from the pdf
    result = local_query(query, context) #  add context to model query
 

    history.append((query, str(result[0])))  # append result to chatInterface history
        

    return  history, ""



def upload_pdf(file):
    '''
    Upload a PDF
    Split into chunks 
    Encode each chunk into embeddings
    Assign a unique ID for each chunk embedding
    Construct Chroma DB
    Update your global Chroma DB collection
    '''
    try:
        if file is not None: 

            global collection
            
            file_name = file.name 

            #  Upload pdf document
            loader = PDFMinerLoader(file_name)
            doc = loader.load()

            #  extract chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10, length_function = len)
            texts = text_splitter.split_documents(doc)
        
            texts = [i.page_content for i in texts]

            #  find embedding for each chunk
            doc_emb = st_model.encode(texts)
            doc_emb = doc_emb.tolist()

            #  index the embeddings
            ids = [str(uuid.uuid1()) for _ in doc_emb]
        
            #  add each chunk embedding to ChromaDB
            collection.add(
                embeddings=doc_emb,
                documents=texts,
                ids=ids
            )

    
            return 'Successfully uploaded!'
        else:
            return "No file uploaded."

    except Exception as e:
        return f"An error occurred: {e}"



    
 
with gr.Blocks() as demo:  
    '''
    Frontend for our tool
    '''

    #  Upload a PDF focument
    btn = gr.UploadButton("Upload a PDF", file_types=[".pdf"])
    output = gr.Textbox(label="Output Box") #  to put message indicating the status of upload
    chatbot = gr.Chatbot(height=240) #  our chatbot interface
    
    with gr.Row():
        with gr.Column(scale=0.70):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Type a question",
            ) 


    # Backend for our tool
    # Event handlers
    btn.upload(fn=upload_pdf, inputs=[btn], outputs=[output])
    txt.submit(run_query, [chatbot, txt], [chatbot, txt])


gr.close_all()
demo.queue().launch() # use query for a better performance
