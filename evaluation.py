import gensim
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

def evaluer_modeles_lda(df, id2word, corpus, start=2, limit=12, step=2):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaMulticore(
            corpus=corpus,
            id2word=id2word,
            num_topics=num_topics,
            random_state=100,
            chunksize=100,
            passes=10,
            alpha='symmetric', # On laisse Gensim gérer l'optimisation de Alpha
            eta=0.1,           # Notre hyperparamètre Beta
            workers=3
        )
        model_list.append(model)
        # Calcul du score de cohérence
        coherencemodel = CoherenceModel(
            model=model, 
            texts=df["tokens_processed"], # Les textes tokenisés (important pour C_v)
            dictionary=id2word, 
            coherence='c_v'
        )
        score = coherencemodel.get_coherence()
        coherence_values.append(score)
        
        print(f" -> Score de cohérence pour K={num_topics} : {score:.4f}\n")

    return model_list, coherence_values

def tracer_graphique_coherence(start, limit, step, coherence_values):
    x = range(start, limit, step)
    plt.figure(figsize=(10, 6))
    plt.plot(x, coherence_values, marker='o', linestyle='-', color='b')
    plt.xlabel("Nombre de Sujets (K)")
    plt.ylabel("Score de Cohérence (C_v)")
    plt.title("Évaluation des modèles LDA : Recherche du meilleur K")
    plt.xticks(x)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Sauvegarde du graphique en image
    plt.savefig("graphique_evaluation.png")
    #print("\nLe graphique a été sauvegardé sous le nom 'graphique_evaluation.png'")
    plt.show()