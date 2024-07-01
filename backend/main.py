import json
import math
import random
from typing import List

import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

############################################## MODEL ################################################
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.db_eric import Papers
from utils import *

faiss_index = faiss.read_index("database/eric_index.index")

classifier = RandomForestClassifier(n_estimators=100, random_state=42)

current_suggestions = []


with open ('database/eric_counts.json') as f:
    frequency_dict = json.load(f)

db_url = "sqlite:///database/eric_database.db"
engine = create_engine(db_url)
Session = sessionmaker(bind=engine)
app = FastAPI()

@app.get("/data")
async def root(skip: int = 0, limit: int = 20, start_year: int = 1950, end_year: int = 2023):
    session = Session()
    data = session.query(Papers).filter(Papers.publication_year >= start_year, Papers.publication_year <= end_year).order_by(Papers.counts.desc()).offset(skip).limit(limit).all()
    session.close()
    return {"data": data}
class CountRequest(BaseModel):
    subject_list: List[str]
    start_year: str = 1950
    end_year: str = 2023
# Define the endpoint
@app.post("/count")
async def count(request: CountRequest):
    res = {}
    for subject in request.subject_list:
        count = getCount(frequency_dict, subject, request.start_year, request.end_year)
        res[subject] = count
    return res


class TrainRequest(BaseModel):
    liked: List[int]

@app.post("/trainModel")
async def trainModel(request: TrainRequest):
    global current_suggestions
    
    # Reconstruct embeddings for liked and not liked papers
    liked_embeddings = [faiss_index.reconstruct(i) for i in request.liked]
    not_liked_embeddings = [faiss_index.reconstruct(i) for i in range(faiss_index.ntotal) if i not in request.liked]
    n_not_liked = 1000
    random_not_liked = random.sample(not_liked_embeddings, n_not_liked)
    # Combine embeddings and labels
    X = liked_embeddings + random_not_liked
    y = [1] * len(liked_embeddings) + [0] * len(random_not_liked)
    
    # Train the classifier
    classifier.fit(X, y)
    
    # Reconstruct all embeddings for scoring
    all_embeddings = [faiss_index.reconstruct(i) for i in range(faiss_index.ntotal)]
    
    # Predict scores for all embeddings
    scores = classifier.predict_proba(all_embeddings)[:, 1]
    
    # Create suggestions with scores
    current_suggestions = [(i, score) for i, score in enumerate(scores)]
    
    # Sort suggestions by score in descending order
    current_suggestions.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top 10 suggestions
    return {"message": "Model trained successfully", "suggestions": current_suggestions[:10]}




@app.get("/getSuggestions")
async def get_suggestions(skip: int = 0, limit: int = 20, start_year: int = 1950, end_year: int = 2023):
    global current_suggestions
    if not current_suggestions:
        raise HTTPException(status_code=400, detail="No suggestions available. Please train the model first.")
    
    session = Session()
    suggestion_ids = [i[0] for i in current_suggestions]
    
    # Retrieve data from the database
    suggested_data = session.query(Papers).filter(
        Papers.publication_year >= start_year,
        Papers.publication_year <= end_year,
        Papers.id.in_(suggestion_ids)
    ).all()
    
    if not suggested_data:
        raise HTTPException(status_code=404, detail="No suggestions found for the given criteria.")
    
    # Create a mapping from suggestion ID to its score
    id_to_score = {i[0]: i[1] for i in current_suggestions}
    
    # Sort the data based on the order in current_suggestions and attach scores
    suggested_data_sorted = sorted(suggested_data, key=lambda x: id_to_score[x.id], reverse=True)
    
    # Apply pagination
    suggested_data_paginated = suggested_data_sorted[skip:skip + limit]
    
    # Attach scores to the results
    results_with_scores = [
        {
            "id": paper.id,
            "title": paper.title,
            "author": paper.author,
            "description": paper.description,
            "subject": paper.subject,
            "publication_year": paper.publication_year,
            "counts": paper.counts,
            "score": id_to_score[paper.id]
        }
        for paper in suggested_data_paginated
    ]
    
    session.close()
    return {"data": results_with_scores}

class ClusterRequest(BaseModel):
    topics: List[str]
    n_paper: int

@app.post("/cluster")
async def cluster(request: ClusterRequest):
    global faiss_index
    idxs = []
    models = {}
    similarities = {}
    session = Session()
    for topic in request.topics:
        classifier = RandomForestClassifier(n_estimators=64, random_state=42)
        papers = session.query(Papers).filter(
            Papers.subject.ilike(f'%{topic}%')
        ).all()
        if papers == []:
            continue
        ids = [paper.id for paper in papers]
        ids = random.sample(ids, min(len(ids), 100))
        x = [faiss_index.reconstruct(i) for i in ids] + random.sample([faiss_index.reconstruct(i) for i in range(faiss_index.ntotal) if i not in ids], len(ids))
        y = [1] * len(ids) + [0] * (len(ids))
        classifier.fit(x, y)
        year_indexes = session.query(Papers).filter(Papers.publication_year >= 1950, Papers.publication_year <= 2023).all()
        scores = classifier.predict_proba([faiss_index.reconstruct(i) for i in range(faiss_index.ntotal)])[:, 1]
        suggestions = [(i, score) for i, score in enumerate(scores)]
        suggestions.sort(key=lambda x: x[1], reverse=True)
        paper_each = request.n_paper
        idxs.append([i[0] for i in suggestions[:paper_each]])
        models[topic] = classifier
    for topic, model in models.items():
        for idx_list in idxs:  # Iterate over lists of indices
            for idx in idx_list:
                paper = session.query(Papers).filter(Papers.id == idx).first()
                title = paper.title
                author = paper.author
                title_author = title + "###" + author
                if title_author not in similarities:
                    similarities[title_author] = {}  # Initialize the dictionary for this index if it doesn't exist
                similarity_score = (model.predict_proba([faiss_index.reconstruct(idx)])[:, 1][0])
                # adjusted_similarity_score = math.log(1 + similarity_score)
                similarities[title_author][topic] = similarity_score
    session.close()
    return similarities

    