import spacy , nltk , pdfplumber , matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
nltk.download('stopwords');nltk.download('punkt')

#print(stopwords.words('arabic')[:50])

def pars_pdf(file_path):
    doc = ""
    with pdfplumber.open(file_path) as pdf:
        for pg in pdf.pages:
            doc += pg.extract_text() or""
    return doc
doc = pars_pdf(r"C:\SDA\M3\NLP\tp2\rapport-lda.pdf")
#print(doc[:800])

wordcloud = WordCloud(background_color="white", max_words=400).generate(doc)
plt.imshow(wordcloud);plt.axis("off");plt.show()
stop_words = set(stopwords.words('french'))

word_tokens = word_tokenize(doc)
filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
filtered_doc = " ".join(filtered_sentence)

wordcloud2 = WordCloud(background_color="white", max_words=400).generate(filtered_doc)
plt.imshow(wordcloud2);plt.axis("off");plt.show()

word_tokens = word_tokenize(filtered_doc)
filtered_sentence_no_alpha = "".join([w for w in word_tokens if w.isalpha()])

nlp = spacy.load('fr_core_news_md')
text = nlp(filtered_sentence_no_alpha)
text_after_lemm = "".join([t.lemma_ for t in text])

wordcloud3 = WordCloud(background_color="white", max_words=400).generate(text_after_lemm)
plt.imshow(wordcloud3);plt.axis("off");plt.show()

#Count
cv = CountVectorizer()
X = cv.fit_transform([text_after_lemm])
print(cv.get_feature_names_out()[:30])
pd.DataFrame(X.toarray(), columns=cv.get_feature_names_out()).iloc[:, :20]

#TF_IDF
tfidf = TfidfVectorizer()
Xt = tfidf.fit_transform([text_after_lemm])
print(tfidf.get_feature_names_out()[:30])