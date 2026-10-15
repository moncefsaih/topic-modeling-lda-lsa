from data import get_data
from pretraitement import preprocess_data
from lda_entrainement import train_lda_model

if __name__ == "__main__":
    # Étape 1 : Collecte et préparation du corpus
    df = get_data()

    # Étape 2 : Prétraitement des données
    df, id2word = preprocess_data(df)

    # Étape 3 : Entraînement du modèle LDA
    
    lda_model = train_lda_model(df, id2word)