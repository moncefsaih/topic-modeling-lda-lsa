from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

def identity(doc):
    return doc  # les documents sont déjà des listes de tokens

def create_tfidf_vectorizer(id2word):
    return TfidfVectorizer(
        analyzer=identity,
        vocabulary=list(id2word.token2id.keys()),
    )

def train_lsa_model(tfidf_matrix, num_topics=4):
    if tfidf_matrix.shape[1] < 2:
        raise ValueError("LSA requires at least two terms in the TF-IDF vocabulary.")
    num_topics = min(num_topics, tfidf_matrix.shape[1] - 1)
    model = TruncatedSVD(n_components=num_topics, random_state=100)
    document_topic_matrix = model.fit_transform(tfidf_matrix)
    return model, document_topic_matrix

def get_top_terms(model, feature_names, num_words=10):
    topics = []
    for component in model.components_:
        top_indices = component.argsort()[-num_words:][::-1]
        topics.append([feature_names[i] for i in top_indices])
    return topics