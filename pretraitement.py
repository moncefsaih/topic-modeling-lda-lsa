import re
import gensim
import gensim.corpora as corpora
from gensim.models.phrases import Phrases, Phraser
import spacy

# Chargement du modèle spaCy (désactivation des composants inutiles pour accélérer)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# =========================================================================
# ÉTAPE 2 : PRÉTRAITEMENT DU TEXTE
# =========================================================================

# 1. Fonctions de nettoyage (minuscules, URLs, balises HTML, chiffres, ponctuation)
def clean_text(text):
    text = text.lower()  # Mise en minuscules
    text = re.sub(
        r"https?://\S+|www\.\S+", "", text
    )  # Suppression des URLs
    text = re.sub(r"<.*?>", "", text)  # Suppression des balises HTML
    text = re.sub(
        r"[^a-zA-Z\s]", "", text
    )  # Suppression des chiffres et ponctuation
    text = re.sub(r"\s+", " ", text).strip()  # Nettoyage des espaces multiples
    return text


# print("--- 1. Nettoyage initial et mise en minuscules ---")
# df["cleaned_text"] = df["text"].apply(clean_text)


# 2. Tokenisation, suppression des stop-words et Lemmatisation (spaCy)
def lemmatize_and_tokenize(texts):
    tokens_list = []
    # Traitement par lots (pipe) pour optimiser les performances
    for doc in nlp.pipe(texts, batch_size=500):
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop  # Suppression des stop-words
            and token.is_alpha  # Conservation des caractères alphabétiques uniquement
            and len(token.lemma_) > 2  # Elimination des mots très courts
        ]
        tokens_list.append(tokens)
    return tokens_list


# print("--- 2. Tokenisation, Stop-words et Lemmatisation ---")
# docs_tokens = lemmatize_and_tokenize(df["cleaned_text"])


# # 3. Détection des n-grammes (Bigrammes et Trigrammes)
# print("--- 3. Détection des bigrammes et trigrammes ---")
# # Construction des modèles de n-grammes avec Gensim
# bigram_words = Phrases(docs_tokens, min_count=5, threshold=10)
# trigram_words = Phrases(bigram_words[docs_tokens], threshold=10)

# bigram_mod = Phraser(bigram_words)
# trigram_mod = Phraser(trigram_words)

# # Application des n-grammes sur le corpus
# docs_ngrams = [trigram_mod[bigram_mod[doc]] for doc in docs_tokens]
# df["tokens_processed"] = docs_ngrams


# # 4. Création du dictionnaire et filtrage des mots (trop rares / trop fréquents)
# print("--- 4. Création du dictionnaire et filtrage fréquentiel ---")
# id2word = corpora.Dictionary(df["tokens_processed"])

# print(f"Taille du vocabulaire avant filtrage : {len(id2word)}")

# # Filtrage : absents dans moins de 5 documents ou présents dans plus de 80% des documents
# id2word.filter_extremes(no_below=5, no_above=0.80)

# print(f"Taille du vocabulaire après filtrage : {len(id2word)}")

# print("\n✓ Étape 2 terminée avec succès !")
# print("\nExemple de tokens obtenus pour le premier document :")
# print(df["tokens_processed"].iloc[0][:15])


def preprocess_data(df):
    df["cleaned_text"] = df["text"].apply(clean_text)

    docs_tokens = lemmatize_and_tokenize(df["cleaned_text"])

    # Construction des modèles de n-grammes avec Gensim
    bigram_words = Phrases(docs_tokens, min_count=5, threshold=10)
    trigram_words = Phrases(bigram_words[docs_tokens], threshold=10)

    bigram_mod = Phraser(bigram_words)
    trigram_mod = Phraser(trigram_words)

    # Application des n-grammes sur le corpus
    docs_ngrams = [trigram_mod[bigram_mod[doc]] for doc in docs_tokens]
    df["tokens_processed"] = docs_ngrams

    id2word = corpora.Dictionary(df["tokens_processed"])

    id2word.filter_extremes(no_below=5, no_above=0.80)
    return df , id2word