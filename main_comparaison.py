import pandas as pd
import gensim
from data import get_data
from preprocess import preprocess_data
from lsa_train import create_tfidf_vectorizer, train_lsa_model
from comparaison import (topics_dominants_lda, topics_dominants_lsa,
                         evaluer_classification)

if __name__ == "__main__":
    K = 4  # même K pour les deux modèles (4 catégories)

    df = get_data()
    df, id2word = preprocess_data(df)
    y_true = df["category"].to_numpy()
    corpus = [id2word.doc2bow(doc) for doc in df["tokens_processed"]]

    # --- LDA ---
    lda = gensim.models.LdaMulticore(
        corpus=corpus, id2word=id2word, num_topics=K,
        random_state=100, passes=10, alpha="symmetric", eta=0.1, workers=3)
    topics_lda = topics_dominants_lda(lda, corpus)
    res_lda = evaluer_classification(topics_lda, y_true, "LDA")

    # --- LSA ---
    vectorizer = create_tfidf_vectorizer(id2word)
    tfidf_matrix = vectorizer.fit_transform(df["tokens_processed"])
    lsa, doc_topics = train_lsa_model(tfidf_matrix, num_topics=K)
    topics_lsa = topics_dominants_lsa(doc_topics)
    res_lsa = evaluer_classification(topics_lsa, y_true, "LSA")

    # --- Tableau comparatif ---
    tableau = pd.DataFrame([res_lda, res_lsa]).round(4)
    print("\n", tableau.to_string(index=False))
    tableau.to_csv("comparaison_lda_lsa.csv", index=False)