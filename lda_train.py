import gensim

def train_lda_model(df, id2word , NUM_TOPICS = 4):

    # ÉTAPE 3 : VECTORISATION (Bag of Words)

    # Création du corpus mathématique
    # Pour chaque document, on compte la fréquence de chaque ID du dictionnaire
    corpus = [id2word.doc2bow(doc) for doc in df["tokens_processed"]]

    
    # ÉTAPE 4 : ENTRAÎNEMENT DU MODÈLE LDA

    # LdaMulticore est plus rapide que LdaModel classique sur les gros volumes
    lda_model = gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=NUM_TOPICS,
        random_state=100,
        chunksize=100,        # Nombre de documents lus à la fois en mémoire
        passes=10,            # Nombre d'allers-retours sur l'ensemble du corpus
        alpha='symmetric',    # Laisse Gensim gérer l'hyperparamètre Alpha
        eta=0.1,              # Notre hyperparamètre Beta
        workers=3             # Nombre de cœurs CPU à utiliser (à ajuster selon votre PC)
    )

    return lda_model,corpus

# # =========================================================================
# # ÉTAPE 3 : VECTORISATION (Bag of Words)
# # =========================================================================
# print("--- 5. Vectorisation du corpus (doc2bow) ---")

# # Création du corpus mathématique
# # Pour chaque document, on compte la fréquence de chaque ID du dictionnaire
# corpus = [id2word.doc2bow(doc) for doc in df["tokens_processed"]]

# print(f"Nombre de documents vectorisés : {len(corpus)}")
# print("Aperçu du Doc 1 vectorisé (ID_mot, Fréquence) :", corpus[0][:10])


# # =========================================================================
# # ÉTAPE 4 : ENTRAÎNEMENT DU MODÈLE LDA
# # =========================================================================
# print("\n--- 6. Entraînement du modèle LDA ---")

# # Définition du nombre de sujets (On en cherche 4 car on a 4 catégories au départ)
# NUM_TOPICS = 4 

# # LdaMulticore est plus rapide que LdaModel classique sur les gros volumes
# lda_model = gensim.models.LdaMulticore(
#     corpus=corpus,
#     id2word=id2word,
#     num_topics=NUM_TOPICS,
#     random_state=100,
#     chunksize=100,        # Nombre de documents lus à la fois en mémoire
#     passes=10,            # Nombre d'allers-retours sur l'ensemble du corpus
#     alpha='symmetric',    # Laisse Gensim gérer l'hyperparamètre Alpha
#     eta=0.1,              # Notre hyperparamètre Beta
#     workers=3             # Nombre de cœurs CPU à utiliser (à ajuster selon votre PC)
# )

# print("\n✓ Modèle LDA entraîné avec succès !")

# # Affichage des mots-clés pour chaque sujet
# print("\n--- MATRICE PHI (Composition des sujets) ---")
# for idx, topic in lda_model.print_topics(-1):
#     print(f"Sujet {idx} : {topic}\n")


# # =========================================================================
# # ÉTAPE 5 : ÉVALUATION DU MODÈLE
# # =========================================================================
# print("\n--- 7. Évaluation (Score de Cohérence) ---")

# # Calcul de la cohérence C_V (plus c'est proche de 1, plus les sujets sont lisibles par un humain)
# coherence_model_lda = CoherenceModel(
#     model=lda_model, 
#     texts=df["tokens_processed"], 
#     dictionary=id2word, 
#     coherence='c_v'
# )
# coherence_lda = coherence_model_lda.get_coherence()
# print(f"Score de cohérence (C_V) du modèle : {coherence_lda:.4f}")

