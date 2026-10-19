import joblib
from preprocess import preprocess_texte
from lsa_train import identity  # nécessaire pour recharger le vectoriseur

lsa = joblib.load("lsa_final.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

def predire_topic_lsa(texte):
    tokens = preprocess_texte(texte)
    X = vectorizer.transform([tokens])
    if X.nnz == 0:
        return None
    scores = lsa.transform(X)[0]
    return sorted(enumerate(scores), key=lambda x: abs(x[1]), reverse=True)

texte = "NASA launched a new rocket to the moon last night"
for topic_id, score in predire_topic_lsa(texte):
    print(f"Topic {topic_id} : {score:.3f}")