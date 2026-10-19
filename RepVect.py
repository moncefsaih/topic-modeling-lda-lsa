import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================================================================
# ÉTAPE 3 : REPRÉSENTATION VECTORIELLE
# =========================================================================

# 1. Préparation pour LDA : Matrice Document-Terme en Bag of Words (BoW)
# Transformation de chaque document en vecteur de fréquences (id_mot, compte)
corpus_bow = [
    id2word.doc2bow(doc) for doc in df["tokens_processed"]
]

# print(f"Nombre de documents représentés en BoW : {len(corpus_bow)}")
# print("Exemple de représentation BoW pour le 1er document (id_mot, frequence) :")
# print(corpus_bow[0][:10])


# 2. Préparation pour LSA : Matrice Document-Terme pondérée par TF-IDF
# print("\n--- 2. Représentation TF-IDF (pour LSA) ---")

# # Reconstruction des chaînes de caractères à partir des jetons pré-traités
# raw_processed_strings = [" ".join(doc) for doc in df["tokens_processed"]]

# # Utilisation du vocabulaire filtré à l'étape 2 pour garantir la cohérence
# vocab_list = [id2word[i] for i in sorted(id2word.keys())]

# # Instanciation du vectoriseur TF-IDF
# tfidf_vectorizer = TfidfVectorizer(
#     vocabulary=vocab_list, lowercase=False
# )
# tfidf_matrix = tfidf_vectorizer.fit_transform(raw_processed_strings)

# print(
#     f"Dimensions de la matrice TF-IDF (Documents x Vocabulaire) : {tfidf_matrix.shape}"
# )

# print("\n✓ Étape 3 terminée avec succès !")