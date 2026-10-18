import pandas as pd
import numpy as np
from langdetect import DetectorFactory, detect
from sklearn.datasets import fetch_20newsgroups

# Fixer la graine pour obtenir des résultats de détection reproductibles
DetectorFactory.seed = 0


# Fonction de détection sécurisée de la langue
def detect_lang_safe(text):
    try:
        return detect(text)
    except:
        # En cas d'erreur (ex. texte contenant uniquement des symboles ou nombres)
        return "unknown"

# ÉTAPE 1 : COLLECTE ET PRÉPARATION DU CORPUS
def get_data():
    categories = ['rec.autos', 'sci.space', 'rec.sport.baseball', 'talk.politics.mideast']
    raw_dataset = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

    df = pd.DataFrame({
        'doc_id': range(len(raw_dataset.data)),
        'text': raw_dataset.data
    })

    # Nettoyage des espaces blancs de début/fin
    df['text'] = df['text'].astype(str).str.strip()

    # Suppression des chaînes vides
    df = df[df['text'] != '']

    # Suppression des doublons stricts
    df = df.drop_duplicates(subset=['text'])

    # Calcul de la longueur en mots et filtrage (< 5 mots)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    MIN_WORDS = 5
    df = df[df['word_count'] >= MIN_WORDS].reset_index(drop=True)

    # Normalisation UTF-8
    df['text'] = df['text'].apply(lambda x: x.encode('utf-8', errors='ignore').decode('utf-8'))

    # Détection de la langue de chaque document
    df['language'] = df['text'].apply(detect_lang_safe)
    return df










# # =========================================================================
# # ÉTAPE 1 : COLLECTE ET PRÉPARATION DU CORPUS
# # =========================================================================

# # 1. Rassembler les documents
# print("--- 1. Rassemblement des documents ---")
# categories = ['rec.autos', 'sci.space', 'rec.sport.baseball', 'talk.politics.mideast']
# raw_dataset = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

# df = pd.DataFrame({
#     'doc_id': range(len(raw_dataset.data)),
#     'text': raw_dataset.data
# })
# #print(f"Nombre de documents bruts rassemblés : {len(df)}")

# # 2. Supprimer les doublons et les documents vides ou trop courts
# #print("\n--- 2. Nettoyage : doublons et documents courts ---")

# # Nettoyage des espaces blancs de début/fin
# df['text'] = df['text'].astype(str).str.strip()

# # Suppression des chaînes vides
# df = df[df['text'] != '']

# # Suppression des doublons stricts
# df = df.drop_duplicates(subset=['text'])

# # Calcul de la longueur en mots et filtrage (< 5 mots)
# df['word_count'] = df['text'].apply(lambda x: len(x.split()))
# MIN_WORDS = 5
# df = df[df['word_count'] >= MIN_WORDS].reset_index(drop=True)

# #print(f"Nombre de documents retenus après filtrage : {len(df)}")

# # 3. Vérifier la langue et l'encodage
# #print("\n--- 3. Normalisation de l'encodage ---")
# # Normalisation UTF-8
# df['text'] = df['text'].apply(lambda x: x.encode('utf-8', errors='ignore').decode('utf-8'))

# # Détection de la langue de chaque document
# df['language'] = df['text'].apply(detect_lang_safe)

# print(df)
# #print("\n✓ Étape 1 terminée avec succès !")
# #print(df[['doc_id', 'word_count', 'text']].head(3))