from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
docs = ["prix alimentation hausse" , "prix energie hausse" , "football victoire equipe"]
tfidf = TfidfVectorizer(ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(docs)
print(tfidf.get_feature_names_out())
print(X_tfidf.toarray())
#similarites = cosine_similarity(X_tfidf)
#print(similarites.round(3))

requete = ["hausse du prix de carburant"]