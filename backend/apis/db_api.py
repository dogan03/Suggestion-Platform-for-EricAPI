import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from scripts.counter import FrequencyCounter
from scripts.faiss_ctrl import FAISS

faiss_controller = FAISS.load("faiss_db_1000.pkl")
counter_controller = FrequencyCounter.load("frequency_counter.pkl")
app = FastAPI()
from sentence_transformers import SentenceTransformer

from scripts.svm import SVM

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = SVM(encoder=encoder)
app = FastAPI()


class QueryRequest(BaseModel):
    query_txt: str
    k: int


@app.post("/query")
def queryDb(request: QueryRequest):
    try:
        res = faiss_controller.query(request.query_txt, request.k)
        return res
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid input.")


@app.get("/data")
def getData():
    try:
        return faiss_controller.data
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid input.")


class FrequencyRequest(BaseModel):
    start_date: int
    end_date: int


@app.post("/get_frequencies")
def getFrequency(request: FrequencyRequest):
    start_date = request.start_date
    end_date = request.end_date
    return counter_controller.getIntervalCounts(start_date, end_date)


class QueryRequestTrain(BaseModel):
    liked: list
    not_liked: list


@app.post("/train")
def queryTrain(request: QueryRequestTrain):
    try:
        liked_emb = [faiss_controller.index.reconstruct(like) for like in request.liked]
        not_liked_emb = [
            faiss_controller.index.reconstruct(not_like)
            for not_like in request.not_liked
        ]
        model.trainSVM(liked_emb, not_liked_emb)

    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid input.")


@app.get("/get_logit")
def querryLogit(idx: int):
    try:
        embed = faiss_controller.index.reconstruct(idx)
        return {
            "logit": model.getLogit(embed)
        }  # Convert logits to list before returning
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid input.")


## Add get embedd.

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
