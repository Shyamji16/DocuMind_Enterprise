from fastapi import FastAPI, UploadFile
from app.ingestion import load_document
from app.retrieval import store_embeddings, generate_answer
import shutil

app = FastAPI()

@app.get("/")
def home():
    return {"message": "DocuMind AI is running"}

@app.post("/upload")
async def upload(file: UploadFile):

    path = f"docs/{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks = load_document(path)

    store_embeddings(chunks)

    return {"message": "Document uploaded successfully"}

@app.get("/ask")
def ask(query: str):

    answer = generate_answer(query)

    return {"answer": answer}