from gensim.models import LdaMulticore
from gensim.corpora import Dictionary
from preprocess import preprocess_texte

lda = LdaMulticore.load("lda_final.model")
id2word = Dictionary.load("dictionnaire.dict")

def predire_topic(texte):
    tokens = preprocess_texte(texte)
    bow = id2word.doc2bow(tokens)
    if not bow:
        return None
    topics = lda.get_document_topics(bow, minimum_probability=0)
    return sorted(topics, key=lambda x: x[1], reverse=True)

texte = "NASA launched a new rocket to the moon last night"
for topic_id, proba in predire_topic(texte):
    print(f"Topic {topic_id} : {proba:.2%}")