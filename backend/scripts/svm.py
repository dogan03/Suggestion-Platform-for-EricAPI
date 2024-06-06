import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


class SVM:
    def __init__(self, encoder) -> None:
        self.encoder = encoder
        self.clf = None

    def trainSVM(self, liked_emb, not_liked_emb):
        # Concatenate the arrays
        x = np.concatenate((liked_emb, not_liked_emb), axis=0)
        # Create labels
        y = np.concatenate(
            (np.ones(len(liked_emb)), np.zeros(len(not_liked_emb))), axis=0
        )
        # Initialize and train the Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(x, y)

    def getLogit(self, embedding):
        embedding = np.array(embedding).reshape(1, -1)
        # Get the probability estimates for class 1
        proba = self.model.predict_proba(embedding)[0, 1]
        return proba
