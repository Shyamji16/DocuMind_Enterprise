from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from app.config import PINECONE_API_KEY, GROQ_API_KEY, INDEX_NAME
from app.prompts import SYSTEM_PROMPT

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

client = Groq(api_key=GROQ_API_KEY)
def store_embeddings(chunks):

    vectors = []

    for i, chunk in enumerate(chunks):

        vector = embedding_model.embed_query(chunk.page_content)

        vectors.append({
            "id": str(i),
            "values": vector,
            "metadata": {
                "text": chunk.page_content,
                "page": chunk.metadata.get("page",0)
            }
        })

    index.upsert(vectors)
def search(query):

    vector = embedding_model.embed_query(query)

    results = index.query(
        vector=vector,
        top_k=4,
        include_metadata=True
    )

    contexts = []

    for match in results["matches"]:
        contexts.append(match["metadata"]["text"])

    return "\n".join(contexts)
def generate_answer(query):

    context = search(query)

    prompt = f"""
Context:
{context}

Question:
{query}
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":prompt}
        ]
    )

    return completion.choices[0].message.content


