import numpy as np
import matplotlib.pyplot as plt


def afficher_topics_lsa(model, feature_names, num_words=8):
    """Interprétation : mots les plus positifs et négatifs de chaque topic."""
    for idx, component in enumerate(model.components_):
        ordre = np.argsort(component)
        positifs = [(feature_names[i], component[i]) for i in ordre[::-1][:num_words]]
        negatifs = [(feature_names[i], component[i]) for i in ordre[:num_words]]
        print(f"Topic {idx}")
        print("   + :", ", ".join(f"{m} ({p:.2f})" for m, p in positifs))
        print("   - :", ", ".join(f"{m} ({p:.2f})" for m, p in negatifs))


def visualiser_mots_lsa(model, feature_names, num_words=8, fichier="lsa_topics.png"):
    """Visualisation 1 : mots les plus influents par topic."""
    k = model.components_.shape[0]
    cols = 2
    rows = (k + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
    axes = np.array(axes).reshape(-1)
    for idx, component in enumerate(model.components_):
        top = np.argsort(np.abs(component))[::-1][:num_words]
        mots = [feature_names[i] for i in top][::-1]
        poids = component[top][::-1]
        couleurs = ["tab:blue" if p >= 0 else "tab:red" for p in poids]
        axes[idx].barh(mots, poids, color=couleurs)
        axes[idx].set_title(f"Topic {idx}")
    for j in range(k, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(fichier)
    plt.show()


def visualiser_documents_lsa(doc_topics, categories, fichier="lsa_documents.png"):
    """Visualisation 2 : documents projetés sur les 2 premiers axes."""
    categories = np.array(categories)
    plt.figure(figsize=(8, 6))
    for cat in sorted(set(categories)):
        masque = categories == cat
        plt.scatter(doc_topics[masque, 0], doc_topics[masque, 1],
                    s=8, alpha=0.5, label=cat)
    plt.xlabel("Axe latent 0")
    plt.ylabel("Axe latent 1")
    plt.title("Projection des documents dans l'espace LSA")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(fichier)
    plt.show()