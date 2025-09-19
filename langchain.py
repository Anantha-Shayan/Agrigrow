import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
# from langchain.llms import LlamaCpp - for local LLM usage
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory

app = FastAPI(title="LangChain FAISS Chatbot")

# Pydantic request model
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

# Setup: embeddings + vectorstore(FAISS)
EMBED_MODEL = "all-MiniLM-L6-v2"
embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

# ingestion
raw_texts = [
    ("https://example.com/article1", "This is the content of article 1 about crop X..."),
    ("https://example.com/article2", "This text covers market prices and mandi API usage..."),
]
documents = [Document(page_content=txt, metadata={"source": src}) for src, txt in raw_texts]

# split & create FAISS index
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
# Overlapping means that each chunk shares some content with the previous/next chunk.
# This ensures that information at the boundary of one chunk isn’t lost and helps maintain context.
chunks = splitter.split_documents(documents)
vectorstore = FAISS.from_documents(chunks, embeddings)  # builds FAISS index

# Choose LLM backend
# USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "0") == "1"
# if USE_LOCAL_LLM:
#     # Example: if you have llama.cpp + quantized model:
#     # llm = LlamaCpp(model_path=os.getenv("LLAMA_PATH"))
#     raise RuntimeError("Local LLM path not configured in example. Set USE_LOCAL_LLM=0 to use HuggingFaceHub.")
# else:
hf_token = os.getenv("HUGGINGFACEHUB_API_KEY")
if not hf_token:
    raise RuntimeError("Please set HUGGINGFACEHUB_API_KEY for HuggingFaceHub LLM usage.")
llm = HuggingFaceHub(repo_id=os.getenv("HF_REPO", "tiiuae/falcon-7b-instruct"),
                    huggingfacehub_api_token=hf_token)

# Memory (Summary)
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", max_token_limit=500)

# Retrieval augmented generation chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) # retrieve top 4 relevant docs from FAISS after embedding similarity search
qa = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory) # Creates a conversational q&a chain by combining llm for generating answers, retriever for fetching relevant docs, and memory for conversation history

# Prompt template
SYSTEM_INSTRUCTION = "Do not use technical terms in your response. Answer simply, politely and clearly for a farmer."

@app.post("/chat")
def chat(req: ChatRequest):
    if not req.question:
        raise HTTPException(status_code=400, detail="Question empty")
    # build inputs for chain
    # (LangChain will append memory and retrieved docs internally)
    out = qa({"question": f"{SYSTEM_INSTRUCTION}\nUser: {req.question}"})

    answer = out.get("answer") or out.get("text") or ""
    sources = []
    for d in out.get("source_documents", []):
        sources.append(d.metadata.get("source"))
    return {"answer": answer.strip(), "sources": sources}
