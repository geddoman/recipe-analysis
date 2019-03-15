import pandas as pd
import pyLDAvis.gensim
from gensim.models import TfidfModel, LdaModel, LsiModel, coherencemodel
from gensim.corpora import Dictionary
from text_preprocessor import text_prepro


def text_vectorizer(text_series):
    """
    Helper function that and vectorizes the text to gensim's corpus
    format and creates the id2word dictionary.

    Parameters:
    ----------
    text_series : pandas.Series object containing the clean text strings

    Returns:
    -------
    gensim id2word and corpus objects

    """

    text_id2word = Dictionary(documents=text_series.values)
    text_corpus = [text_id2word.doc2bow(doc) for doc in text_series.values]

    return text_id2word, text_corpus


def tfidf_model_maker(text_series):
    """
    Function that applies a TfIdf model on the cleaned text.

    Parameters:
    ----------
    text_series : pandas.Series object containing the clean text strings

    Returns:
    -------
    pandas DataFrame containing the cumulative term frequencies, the document
    frequencies and the tfidf score of the keywords in the text. The DataFrame
    is sorted by descending order of TfIdf

    """

    text_id2word, text_corpus = text_vectorizer(text_series)

    tfidf_model = TfidfModel(corpus=text_corpus)
    tfidf_corpus = tfidf_model[text_corpus]


    keyword_df = pd.DataFrame(index=text_id2word.keys())

    keyword_df['keyword'] = ''
    keyword_df['term_frequency'] = 0
    keyword_df['document_frequency'] = 0
    keyword_df['tfidf'] = 0.0

    for id, word in text_id2word.items():
        keyword_df.at[id, 'keyword'] = word

    for doc in text_corpus:
        for word_id, freq in doc:
            keyword_df.at[word_id, 'term_frequency'] += freq

    for word_id, df in tfidf_model.dfs.items():
        keyword_df.at[word_id, 'document_frequency'] = df

    for doc in tfidf_corpus:
        for word_id, tfidf_score in doc:
            keyword_df.at[word_id, 'tfidf'] += tfidf_score

    keyword_df.sort_values(by='tfidf', ascending=False, inplace=True)

    return keyword_df


def lda_model_maker(text_series, n_topics=20):
    """
    Function that applies an LDA Model on the cleaned text.

    Parameters:
    ----------
    text_series : pandas.Series
        Series containing the clean text strings

    n_topics : int, default 20
        Number of topics returned by the model

    Returns:
    -------
    gensim LDA model from the text and visualization for the model

    """

    text_id2word, text_corpus = text_vectorizer(text_series)

    lda_model = LdaModel(text_corpus,num_topics=n_topics,id2word=text_id2word)
    lda_vis = pyLDAvis.gensim.prepare(lda_model, text_corpus, text_id2word)

    return lda_model, lda_vis