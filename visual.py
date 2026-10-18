import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

def visualize_lda_model(lda_model, id2word, corpus):
    vis_data = gensimvis.prepare(lda_model, corpus, id2word, mds='mmds')
    html_file = 'lda_visualisation.html'
    pyLDAvis.save_html(vis_data, html_file)