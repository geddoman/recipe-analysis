# Importing the necessary libraries
import pandas as pd 
import matplotlib.pyplot as plt
import re
import csv
import pickle
from tqdm import tqdm
from flashtext import KeywordProcessor
from nltk import FreqDist
from nltk.corpus import stopwords
from gensim.models import LdaModel, LsiModel, TfidfModel, CoherenceModel
from gensim.corpora import Dictionary
import spacy


# ____Text Preprocessing Section_____#
"""
Some spices and herbs (From hereon I use the term interchangeably)
can be expressed as 2 or 3 word phrases (such as "black pepper")
they also can have multiple variations (such as "ground black pepper" or 
"crushed black pepper", dismissing the fact that ground and crushed 
black pepper would be considered by many as 2 completely different
spices each with their own use. For my examples and experiments I consider
them the same.

So here we have a simple problem, to achieve better results 
(regardless of the experiment) we want to group all the variations of each
spice or herb under one term. One way of doing so is to choose a term 
(good practice is to choose the most common representation) and replace 
all other variants with the chosen term. Almost all of the algorithms and 
methods we'll use do not detect 2 or 3 word phrases on their own, the 
easiest method is to use an underscore to replace the space in these terms 
so we will have "black_pepper" which to our algorithms is considered
as one word, yet we still preserve the unique combination of those
2 words together.

So my method here is simple. It focuses only on some spices and herbs,
which I have collected from online resources and heavily edited the list
adding the different variants manually, as well as, adding some missing spices
(no onion powder? no dry aromatics??). The text file for spices has each
spice in a line and always starts with the chosen word of the spice
(black_pepper) followed by a tab and then the variants separated by a comma.
The variants should also contain the chosen term (from hereon called the key)
but in its natural form (without underscores).

This use of double delimiters makes it very easy to read this text and have it
stored as a standard python dictionary.

The preprocessing pipeline is targeted for English documents. It uses regex
to remove any unwanted characters and to remove any added spaces. spaCy is 
used for the optional lemmatization (although it is slow), NLTK is used only
to get the English stop word list and finally a small library called flashtext
is used to replace the spice variants.

A very important note here on the preprocessing choices I've made: My goal 
here wasn't optimization, it was breadth of choices and an excercise for me.
My goal also wasn't to have an easy solution, there are premade text 
preprocessing functions in gensim and tons of code available online for 
quick preprocessing solutions. I preferred to get my hands dirty and do it
mostly from scratch.

I believe that text-preprocessing is a problem that has infinite solutions.
In terms of the techniques, libraries used or even the expected outcome.

---> Should add the option to get tagged documents from text to use readily
---> for Doc2Vec models.
"""

# Assigning the replacement dict file path and instantiating the spice_dict

spice_repl_file = 'spice_repl_dict.txt'
spice_dict = {}

# Loading the replacement dictionary and preparing the keyword processor

spice_dict = {}
with open(spice_repl_file) as f:
    spice_row_reader = csv.reader(f,delimiter='\t')
    for key,value in spice_row_reader:
        spice_dict[key] = value.split(',')
    
kp = KeywordProcessor()
kp.add_keywords_from_dict(spice_dict)

# Setting the patterns for regex used in the cleaning helper function

patt = re.compile(r'[\W+]') # Pattern for any non-alphanumeric characters
patt_2 = re.compile(r'\b\d+\b') # Pattern for stand-alone numeric characters
patt_3 = re.compile(r'\b[A-Za-z]\b') # Pattern for single alphabetical characters
patt_4 = re.compile(r' {2,}') # Pattern for repeating whitespace


# Loading the English stop words from NLTK and assiging to a set object

stop_set = set(stopwords.words('english'))

# Updating the stop_set with additional terms
stop_set.update(['-pron-','minute','add','heat','cook','minutes'])

# Loading the spacy model for use in lemmatization

nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])

def text_cleaner(doc):
    """ 

    Helper function that replaces ampersands '&' with 'and' then applies a
    predefined regex pattern to remove special characters, added whitespace
    and returns a lower case document string.

    This function can be used on its own, but it is part of the more
    encompassing text_prepro function.

    Parameters:
    ----------
    doc : str
        The document or string to be cleaned

    Returns:
    -------
    Normalized and lower case str

    """
    doc = doc.replace('&',' and ')
    doc = re.sub(patt,' ',doc)
    doc = re.sub(patt_2,' ',doc)
    doc = re.sub(patt_3,' ',doc)
    doc = re.sub(patt_4,' ',doc)
    return doc.lower().strip()


def lemmatizer(doc):
    """
    
    Helper function for lemmatization of text, it uses spaCy's pre-trained
    model, in this case it is using the standard small model. The results
    include the string -PRON- replacing pronouns in the doc. Due to this
    -PRON- is added to the stop list.


    Parameters:
    ----------
    doc : str
        The document or string to be lemmatized using spaCy

    
    Returns:
    -------
    Str with lemmas of words where applicable

    """
    prepared_doc = nlp(doc)
    return ' '.join([token.lemma_ for token in prepared_doc])


def tokenizer(doc, remove_stop=True):
    """

    Helper function that takes a document string as an argument and tokenizes
    the text into words that are not in the stop list. This should be used as
    the final function after the text is cleaned.


    Parameters:
    ----------
    doc : str
        The document or string of text to be tokenized

    remove_stop : boolean, default True
        Determines whether to remove stop words or not

    
    Returns:
    -------
    List of tokenized words

    """
    if remove_stop:
        return [word.strip() for word in doc.split() if word not in stop_set]
    else:
        return [word.strip() for word in doc.split()]

def text_prepro(text, r_stop=True, lemmatize=True, repl=True):
    """

    Main function that preprocesses the text by:
    1-Replacing ampersands
    2-Removing non-alphanumeric characters and added whitespace
    3-Removing single character words
    4-Removing numerical characters not attached to alphabetical characters
    5-Converting to lowercase
    6-Replacing predefined words/phrases using a custom dictionary
    7-Removing predefined stop words
    8-Tokenizes the text, splitting on horizontal whitespace
    9-Lemmatizes the text

    Returns a cleaned list of tokens from the text.

    Parameters:
    ----------

    text : str
        Document or string of text to be preprocessed

    r_stop : boolean, default True
        Value to pass to the tokenizer to remove or keep stop words

    lemmatize : boolean
        Determines whether to lemmatize the text or not

    Returns:
    -------
    List of cleaned, lemmatized (if chosen), and tokenized words
    
    """

    text_cln = text_cleaner(text)
    if repl:
        text_cln = kp.replace_keywords(text_cln)
    if lemmatize:
        text_cln = lemmatizer(text_cln)
    text_cln = tokenizer(text_cln, remove_stop=r_stop)
    return text_cln


# Loading the recipes as a DataFrame
recipe_file = 'full_recipes.csv'
recipe_df = pd.read_csv(recipe_file, index_col=0)

# Although none would be present, it is best to check for any missing
# Values and handle them as necessary

# Cleaning the text of the recipes using the preprocessing function
recipe_df['method_cln'] = recipe_df['method'].apply(text_prepro)

#____EDA SHOULD GO HERE____#

# Basic plot of the most frequent words
# Horribly inefficient of course
word_list = []

for item in recipe_df['method_cln'].values:
    for word in item:
        word_list.append(word)

freq_dist = FreqDist(word_list)

# Matplotlib stuff
plt.figure(figsize=(20, 10))
freq_dist.plot(50)

# Creating the dictionary and corpus
recipe_id2word = Dictionary(documents=recipe_df['method_cln'].values)
recipe_corpus = [recipe_id2word.doc2bow(recipe) for recipe in recipe_df['method_cln'].values]

recipe_tfidf_model = TfidfModel(corpus=recipe_corpus)
recipe_tfidf_corpus = recipe_tfidf_model[recipe_corpus]

# Descriptive DataFrame that will contain the cumulative stats for the keywords
keyword_df = pd.DataFrame(index=recipe_id2word.keys())

keyword_df['keyword'] = ''
keyword_df['term_frequency'] = 0
keyword_df['document_frequency'] = 0
keyword_df['tfidf'] = 0.0

for id,word in recipe_id2word.items():
    keyword_df.at[id,'keyword'] = word
    
for doc in recipe_corpus:
    for word_id, freq in doc:
        keyword_df.at[word_id,'term_frequency'] += freq
    
for word_id, df in recipe_tfidf.dfs.items():
    keyword_df.at[k,'document_frequency'] = v
    
for doc in recipe_tfidf_corpus:
    for word_id, tfidf_score in doc:
        keyword_df.at[word_id,'tfidf'] += tfidf_score   

# Creating the LDA model
recipe_lda = LdaModel(corpus=recipe_corpus, id2word=recipe_id2word, num_topics=15)

# Creating the LSI model
recipe_lsi = LsiModel(corpus=recipe_corpus, id2word=recipe_id2word, num_topics=15)