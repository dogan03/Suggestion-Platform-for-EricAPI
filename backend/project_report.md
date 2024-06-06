# DistilBERT (Fine-Tuned)

- Got data from the Eric API.
- Used its description part to fine-tune BERT using mlm.
- Trained and tested for some queries. However, the results wasn't good.
- DistilBERT base was better
- Works slow when indexing to faiss.

# Sentence Transformer (all-MiniLM-L6-v2)

- Works well and fast. (bcs only 1 embedding dim)
- Memory usage better in faiss.
