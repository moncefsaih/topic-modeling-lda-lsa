from sklearn.feature_extraction.text import TfidfVectorizer 


docs = ["prix alimentation hausse" , "prix energie hausse" , "football victoire equipe"]
tfidf = TfidfVectorizer(ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(docs)
print(tfidf.get_feature_names_out())
#print(X_tfidf.toarray())