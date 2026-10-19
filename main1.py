import joblib
from data import get_data
from preprocess import preprocess_data
from lsa_train import create_tfidf_vectorizer, get_top_terms
from evaluation_lsa import evaluer_modeles_lsa, tracer_graphique_coherence_lsa
from visual_lsa import afficher_topics_lsa, visualiser_mots_lsa, visualiser_documents_lsa

if __name__ == "__main__":
    df = get_data()
    df, id2word = preprocess_data(df)

    # Vectorisation TF-IDF
    vectorizer = create_tfidf_vectorizer(id2word)
    tfidf_matrix = vectorizer.fit_transform(df["tokens_processed"])
    feature_names = vectorizer.get_feature_names_out()

    # Choix de K
    start, limit, step = 2, 12, 2
    modeles, scores = evaluer_modeles_lsa(df, id2word, tfidf_matrix,
                                          feature_names, start, limit, step)

    i = scores.index(max(scores))
    meilleur_k = list(range(start, limit, step))[i]
    meilleur_modele = modeles[i]
    print(f"Meilleur K (LSA) : {meilleur_k} | c_v = {scores[i]:.4f}")
    tracer_graphique_coherence_lsa(start, limit, step, scores)

    # Interprétation
    for idx, mots in enumerate(get_top_terms(meilleur_modele, feature_names, 8)):
        print(f"Topic {idx} : {mots}")

    # Sauvegarde
    joblib.dump(meilleur_modele, "lsa_final.joblib")
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")

    afficher_topics_lsa(meilleur_modele, feature_names)
    visualiser_mots_lsa(meilleur_modele, feature_names)
    doc_topics = meilleur_modele.transform(tfidf_matrix)
    visualiser_documents_lsa(doc_topics, df["category"].tolist())