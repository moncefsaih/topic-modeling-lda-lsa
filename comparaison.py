import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)


def topics_dominants_lda(lda_model, corpus):
    """Topic le plus probable pour chaque document (LDA)."""
    resultat = []
    for bow in corpus:
        dist = lda_model.get_document_topics(bow, minimum_probability=0)
        resultat.append(max(dist, key=lambda x: x[1])[0])
    return np.array(resultat)


def topics_dominants_lsa(doc_topics):
    """Topic dominant (LSA). Les axes sont standardisés pour que l'axe 0,
    qui capte le vocabulaire général, ne domine pas tous les documents."""
    z = (doc_topics - doc_topics.mean(axis=0)) / doc_topics.std(axis=0)
    return np.argmax(np.abs(z), axis=1)


def mapper_topics_categories(topics, y_true):
    """Associe chaque topic à la catégorie la plus fréquente parmi ses documents."""
    mapping = {}
    for t in np.unique(topics):
        cats = y_true[topics == t]
        mapping[t] = pd.Series(cats).value_counts().idxmax()
    return mapping


def evaluer_classification(topics, y_true, nom):
    mapping = mapper_topics_categories(topics, y_true)
    y_pred = np.array([mapping[t] for t in topics])

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)

    print(f"\n===== {nom} =====")
    print("Association topic -> catégorie :", mapping)
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Matrice de confusion :\n", confusion_matrix(y_true, y_pred))
    return {"Modèle": nom, "Accuracy": acc, "Precision": p,
            "Recall": r, "F1-score": f1}