import pandas as pd
import gensim
from gensim.models import CoherenceModel

def evaluer_alpha_beta(df, id2word, corpus, meilleur_k):
    """
    Teste plusieurs combinaisons de Alpha et Beta pour un K fixe 
    et retourne les résultats sous forme de tableau (DataFrame).
    """
    # Les valeurs à tester
    valeurs_alpha = ['symmetric', 0.01, 0.1, 1.0]
    valeurs_beta = ['symmetric', 0.01, 0.1, 1.0]  # Dans Gensim, Beta s'appelle 'eta'
    
    resultats = []
    total_iterations = len(valeurs_alpha) * len(valeurs_beta)
    compteur = 1
    
    print(f"--- Lancement du Grid Search pour K={meilleur_k} ---")
    print(f"Nombre total de modèles à entraîner : {total_iterations}\n")

    for a in valeurs_alpha:
        for b in valeurs_beta:
            print(f"[{compteur}/{total_iterations}] Test avec Alpha={a} et Beta={b}...")
            
            # Entraînement du modèle
            model = gensim.models.LdaMulticore(
                corpus=corpus,
                id2word=id2word,
                num_topics=meilleur_k,
                random_state=100,
                chunksize=100,
                passes=10,
                alpha=a,
                eta=b,
                workers=3
            )
            
            # Calcul du score
            coherencemodel = CoherenceModel(
                model=model, 
                texts=df["tokens_processed"], 
                dictionary=id2word, 
                coherence='c_v'
            )
            score = coherencemodel.get_coherence()
            
            # Sauvegarde du résultat
            resultats.append({
                'Alpha': str(a),
                'Beta': str(b),
                'Coherence_Cv': round(score, 4)
            })
            compteur += 1

    # Conversion en tableau lisible
    df_resultats = pd.DataFrame(resultats)
    
    # Tri par ordre décroissant (le meilleur score en premier)
    df_resultats = df_resultats.sort_values(by='Coherence_Cv', ascending=False).reset_index(drop=True)
    
    return df_resultats