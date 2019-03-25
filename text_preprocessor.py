# Importing the necessary libraries

import re
import csv
import pickle
from flashtext import KeywordProcessor
import spacy
import en_core_web_sm


# Setting the patterns for regex used in the cleaning helper function

patt = re.compile(r'[\W+]') # Pattern for any non-alphanumeric characters
patt_2 = re.compile(r'\b\d+\b') # Pattern for stand-alone numeric characters
patt_3 = re.compile(r'\b[A-Za-z]\b') # Pattern for single alphabetical characters
patt_4 = re.compile(r' {2,}') # Pattern for repeating whitespace


# Loading the English stop words from disk and assigining to a set object
stop_set = set()

with open('stop_list.txt',mode='r', encoding='UTF-8') as f_in:
    for line in f_in:
        stop_set.add(line.replace('\n',''))

# Updating the stop_set with additional terms
stop_set.update(['-PRON-'])

# Loading the spacy model for use in lemmatization

nlp = en_core_web_sm.load(disable=['parser','ner'])

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


def make_repl_dict(repl_file):
    """
    Helper function that uses a custom replacement dictionary and returns
    a FlashText KeywordProcessor object to use for the actual replacement.

    Parameters:
    ----------

    repl_file : str
        Path to the replacement dictionary txt file. The file contents should
        be all in lower case and it should contain the key at the start of each
        line with underscores replacing the spaces if present. Then a single
        tab character followed by the values that you want to replace with 
        the key separated by commas. Check the example below.

        key_word_1/tvalue 1,value 2,value 3
        key_word_2/tvalue 4,value 5,value 6

    Returns:
    -------
    FlashText KeywordProcessor object containing the replacement dict.

    """

    repl_dict = {}
    with open(repl_file) as f:
        row_reader = csv.reader(f,delimiter='\t')
        for key,value in row_reader:
            repl_dict[key] = value.split(',')
        
    kp = KeywordProcessor()
    kp.add_keywords_from_dict(repl_dict)

    return kp


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


def tokenizer(doc, remove_stop=True, additional_stop=None):
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
    
    additional_stop : str, Optional
        Additional words to add to the stop list. These must be separated by a
        single comma without any spacing and only single tokens.
        For example: 'stop1,stop2,stop3,stop4'

    
    Returns:
    -------
    List of tokenized words

    """
    if additional_stop != None:
        stop_set.update(additional_stop.split(','))
    if remove_stop:
        return [word.strip() for word in doc.split() if word not in stop_set]
    else:
        return [word.strip() for word in doc.split()]

def text_prepro(text, replace=None, r_stop=True, add_stop=None, lemmatize=True):
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

    replace : str, default None, optional
        Path to replacement dictionary txt file

    r_stop : boolean, default True
        Value to pass to the tokenizer to remove or keep stop words

    add_stop : str, optional
        Additional stop words to pass to the tokenzier. These must be single
        lowercase words separated by a comma without spacing.

    lemmatize : boolean
        Determines whether to lemmatize the text or not

    Returns:
    -------
    List of cleaned, lemmatized (if chosen), and tokenized words

    """

    text_cln = text_cleaner(text)
    if replace != None:
        kp = make_repl_dict(replace)
        text_cln = kp.replace_keywords(text_cln)
    if lemmatize:
        text_cln = lemmatizer(text_cln)
    if add_stop != None:
        text_cln = tokenizer(text_cln, remove_stop=r_stop, additional_stop=add_stop)
        return text_cln
    else:
        text_cln = tokenizer(text_cln, remove_stop=r_stop)
        return text_cln