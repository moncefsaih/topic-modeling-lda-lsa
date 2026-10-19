from sklearn.feature_extraction.text import CountVectorizer
docs = ["prix alimentation hausse" , "prix energie hausse" , "football victoire equipe"]
v = CountVectorizer()
X = v.fit_transform(docs)
#print(v.get_feature_names_out())
#print(X.toarray())

nv = ["hausse du prix de carburant"]

xnv = v.transform(nv)
print(v.get_feature_names_out())
print(xnv.toarray())