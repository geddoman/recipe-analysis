# Importing the necessary libraries

import re
import pickle
from flashtext import KeywordProcessor
import spacy
import en_core_web_sm

# Assigning the file paths to variables

repl_file = 'DataSets/repl_dict'
stop_file = 'DataSets/stop_list_prop.txt'

# Loading the replacement dictionary and preparing the keyword processor

with open(repl_file,mode='r+b') as f:
    repl_dict = pickle.load(f)
    
for k,v in repl_dict.items():
    repl_dict[k] = [syn for syn in v]
    
kp = KeywordProcessor()
kp.add_keywords_from_dict(repl_dict)

# Setting the patterns for regex used in the below function

patt = re.compile(r'[\W+]')
patt_2 = re.compile(r'\b\d+\b')
patt_3 = re.compile(r' {2,}')


# Loading the custom stop word list as a set object for later use

stop_set = set()

with open(stop_file) as f:
    for word in f:
        stop_set.add(word.strip())

# Loading the spacy model for use in lemmatization
nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])

def text_cleaner(doc):
    """ 

    Helper function that replaces ampersands '&' with 'and' then applies a
    predefined regex pattern to remove special characters, added whitespace
    and returns a lower case document string.

    This function can be used on its own, but it is part of the more
    encompassing text_prepro function.

    """
    doc = doc.replace('&',' and ')
    doc = re.sub(patt,' ',doc)
    doc = re.sub(patt_2,' ',doc)
    doc = re.sub(patt_3,' ',doc)
    return doc.lower().strip()


def lemmatizer(doc):
    """
    
    Helper function for lemmatization of text, it uses spaCy's pre-trained
    model, in this case it is using the standard small model. The results
    include the string -PRON- replacing pronouns in the doc. Due to this
    -PRON- is added to the stop list.

    """
    prepared_doc = nlp(doc)
    return ' '.join([token.lemma_ for token in prepared_doc])


def tokenizer(doc):
    """

    Helper function that takes a document string as an argument and tokenizes
    the text into words that are not in the stop list. This should be used as
    the final function after the text is cleaned.

    Returns a list of tokens (words).

    """
    return [word.strip() for word in doc.split() if word not in stop_set]

def text_prepro(text):
    """

    Main function that preprocesses the text by:
    1-Replacing ampersands
    2-Removing special characters and added whitespace
    3-Converting to lowercase
    4-Replacing predefined words/phrases using a custom dictionary
    5-Removing predefined stop words
    6-Tokenizes the text, splitting on horizontal whitespace

    Returns a cleaned list of tokens from the text.

    """

    text_cln = text_cleaner(text)
    text_cln = kp.replace_keywords(text_cln)
    text_cln = lemmatizer(text_cln)
    text_cln = tokenizer(text_cln)
    return text_cln
