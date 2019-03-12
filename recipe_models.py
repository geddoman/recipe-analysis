#____Model creating and training section____#
"""

The goal of these excercises is to create a topic modeling pipeline for the
recipes using LDA LSI. The resulting topics will probably be of mediocre
quality (as you'll see), however the goal here is not to get the best or most
significant topics, but to create a pipeline that can be further tuned with
better data to get the most significant and relevant topics.
gensim is used for the models. Against gensim's core design ethos, everything
will reside in memory. The number of documents is small (about 2.5K) and as 
a first iteration it will serve the purpose of this task. Admittedly, while
writing this, I realized this is probably a bad decision on my part. Especially
since I know how to write generators in python and not having everything in
memory.

"""



from recipe_text_preprocessor import text_prepro 
from gensim.models import LdaModel, LsiModel, CoherenceModel, TfidfModel
from gensim.corpora import Dictionary
import pandas as pd 
from tqdm import tqdm


# Loading the recipes as a DataFrame
recipe_file = 'full_recipes.csv'
recipe_df = pd.read_csv(recipe_file, index_col=0)

# Cleaning the text of the recipes using the preprocessing function
recipe_df['method_cln'] = recipe_df['method'].apply(text_prepro)

#____EDA SHOULD GO HERE____#

# Basic plot of the most frequent words
# Horribly inefficient ofcourse
word_list = []

for item in recipe_df['method_cln'].values:
    for word in item:
        word_list.append(word)

freq_dist = FreqDist(word_list)

# Matplotlib stuff
plt.figure(figsize=(20,10))
freq_dist.plot(50)

# Creating the dictionary and corpus
recipe_id2word = Dictionary(documents=recipe_df['method_cln'].values)
recipe_corpus = [recipe_id2word.doc2bow(recipe) for recipe in recipe_df['method_cln'].values]

# Creating the LDA model
recipe_lda = LdaModel(corpus=recipe_corpus,id2word=recipe_id2word,num_topics=15)
