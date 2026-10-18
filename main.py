from data import get_data
from preprocess import preprocess_data
from lda_train import train_lda_model
from evaluation import evaluer_modeles_lda
from visual import visualize_lda_model

if __name__ == "__main__":
    # Étape 1 : Collecte et préparation du corpus
    df = get_data()

    # Étape 2 : Prétraitement des données
    df, id2word = preprocess_data(df)

    # Étape 3 : VECTORISATION (Bag of Words) et Entraînement du modèle LDA
    lda_model, corpus = train_lda_model(df, id2word, NUM_TOPICS=4)

    # Étape 4 : Évaluation du modèle LDA
    start,limit,step = 2,12,2
    modeles, scores = evaluer_modeles_lda(df, id2word, corpus, start, limit, step)

    # On trouve l'index du score maximum dans la liste des scores
    meilleur_index = scores.index(max(scores))
        
    # On recrée la liste des K qui ont été testés (2, 4, 6, 8, 10)
    liste_des_k = list(range(start, limit, step))
        
    # On utilise l'index pour extraire les bonnes informations
    meilleur_k = liste_des_k[meilleur_index]
    meilleur_score = scores[meilleur_index]
    meilleur_modele = modeles[meilleur_index] # Le modèle prêt à être utilisé !
    # 1. Sauvegarde
    meilleur_modele.save("lda_final.model") 
    id2word.save("dictionnaire.dict")
    # Étape 5 : Visualisation des résultats
    visualize_lda_model(meilleur_modele, id2word, corpus)
    
    # Étape _ : Affichage du graphique
    #tracer_graphique_coherence(start, limit, step, scores)

    # Etape 5 : Recherche des meilleurs hyperparamètres Alpha et Beta pour le meilleur K
    
    # tableau_scores = evaluer_alpha_beta(df, id2word, corpus, meilleur_k)
    # nom_fichier_csv = "resultats_alpha_beta.csv"
    # tableau_scores.to_csv(nom_fichier_csv, index=False)

    