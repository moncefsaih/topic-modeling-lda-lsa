from gensim.models import CoherenceModel
from lsa_train import train_lsa_model, get_top_terms
import matplotlib.pyplot as plt

def evaluer_modeles_lsa(df, id2word, tfidf_matrix, feature_names,
                        start=2, limit=12, step=2, num_words=10):
    coherence_values, model_list = [], []
    for k in range(start, limit, step):
        model, _ = train_lsa_model(tfidf_matrix, num_topics=k)
        topics = get_top_terms(model, feature_names, num_words)

        cm = CoherenceModel(
            topics=topics,
            texts=df["tokens_processed"],
            dictionary=id2word,
            coherence="c_v",
        )
        score = cm.get_coherence()
        model_list.append(model)
        coherence_values.append(score)
        print(f" -> LSA : cohérence pour K={k} : {score:.4f}")
    return model_list, coherence_values

def tracer_graphique_coherence_lsa(start, limit, step, coherence_values):
    x = range(start, limit, step)
    plt.figure(figsize=(10, 6))
    plt.plot(x, coherence_values, marker='o', linestyle='-', color='r')
    plt.xlabel("Nombre de Sujets (K)")
    plt.ylabel("Score de Cohérence (C_v)")
    plt.title("Évaluation des modèles LSA : Recherche du meilleur K")
    plt.xticks(x)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("graphique_evaluation_lsa.png")
    plt.show()