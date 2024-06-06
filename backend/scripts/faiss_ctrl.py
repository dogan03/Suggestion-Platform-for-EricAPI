import pickle

import faiss
import numpy as np


class FAISS:
    def __init__(self, model) -> None:
        self.model = model
        self.documents = []
        self.index = None
        self.list_of_title = []
        self.list_of_description = []
        self.list_of_subject = []
        self.list_of_year = []

    def init_db(self, eric_df):
        for idx in range(len(eric_df["title"].to_list())):
            if eric_df["description"].to_list()[idx] == str(
                eric_df["description"].to_list()[idx]
            ):
                self.list_of_title.append(eric_df["title"].to_list()[idx])
                self.list_of_description.append(eric_df["description"].to_list()[idx])
                self.list_of_subject.append(eric_df["subject"].to_list()[idx])
                self.list_of_year.append(eric_df["publicationdateyear"].to_list()[idx])
        self.data = {
            "title": self.list_of_title,
            "description": self.list_of_description,
            "subject": self.list_of_subject,
            "year": self.list_of_year,
        }
        embeddings = self.model.encode(self.list_of_description)
        embeddings = np.array(embeddings).astype("float32")
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings.astype("float32")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def query(self, query_text, k, show=True):
        query_embedding = self.model.encode([query_text])
        query_embedding = query_embedding / np.linalg.norm(
            query_embedding, axis=1, keepdims=True
        )
        query_embedding = query_embedding.astype("float32")
        cosine_similarities, indices = self.index.search(query_embedding, k)

        # Convert cosine_similarities and indices to regular Python lists
        cosine_similarities = [
            float(similarity) for similarity in cosine_similarities[0]
        ]
        indices = [int(index) for index in indices[0]]

        distances = [
            (
                1 - similarity
                if similarity < np.finfo(np.float32).max
                else np.finfo(np.float32).max
            )
            for similarity in cosine_similarities
        ]

        res = {
            "distance": [],
            "title": [],
            "subject": [],
            "description": [],
            "publicationdateyear": [],
        }
        for i, (distance, idx) in enumerate(zip(distances, indices)):
            if show:
                print(f"Rank {i+1}:")
                print(f"Document: {self.list_of_description[idx]}")
                print(f"Distance: {distance}")  # This distance will be between 0 and 1
                print()
            res["distance"].append(round(float(distance), 3))
            res["description"].append(self.list_of_description[idx])
            res["title"].append(self.list_of_title[idx])
            res["subject"].append(self.list_of_subject[idx])
            res["publicationdateyear"].append(self.list_of_year[idx])
        return res

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    from scripts.faiss_ctrl import FAISS

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # List of documents (e.g., paper abstracts)
    documents = [
        "I want to read about animals.",
        "goodbye world.",
        "-s in turkish.",
        "morphological improvements.",
    ]

    faiss_ctrl = FAISS(model=model)

    faiss_ctrl.init_db(documents)
    faiss_ctrl.query("this paper is about mammals", 3)
