#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import gensim
import nltk
import re
import string
import timeit as ts
import matplotlib.pyplot as plt
import math 


# In[2]:


wine_data = pd.read_csv("C:\\Users\\Owner\\Documents\\School\\Masters of Science in Analytics\\Semesters\\Fall Semester\\Text Mining\\Kaggle_Wine_File.csv")
reviews = wine_data['description']


# In[4]:


term_vec2 = [ ]

for d in reviews:
    
    #Makes each word in a review lowercase
    d = d.lower()
    
    # Breaks each review down into its individual terms and stores the
    # words in a list where each word is an individual entry
    # EXAMPLE: "hello world" ==> ["hello", "world"]
    term_vec2.append( nltk.TweetTokenizer().tokenize( d ) )


# In[20]:


test = [['hello monther, how are you?'],['great thanks']]

#Tokenize expects a list of lists!
def tokenize(term_vector):
    
    term_tokens = [ ]
    
    for d in term_vector:

        #Makes each word in a review lowercase
        d = d[0].lower()

        # Breaks each review down into its individual terms and stores the
        # words in a list where each word is an individual entry
        # EXAMPLE: "hello world" ==> ["hello", "world"]
        term_tokens.append( nltk.TweetTokenizer().tokenize( d ) )
        
    return term_tokens


# In[6]:


### Remove stop words from the term vectors

#Creates a tokenized list of stop works
stop_words = nltk.corpus.stopwords.words( 'english' )

# Take each tokenized wine review and remove stop words (i.e. only, own, don't, etc)
for i in range( 0, len( term_vec2 ) ):
    term_list2 = [ ]
    
    #Append terms that are not in stop word library into a new wine review vector
    for term in term_vec2[ i ]:
        if term not in stop_words:
            term_list2.append( term )
    
    #Replace the original wine review with the new wine review that doesn't contain stop words
    term_vec2[ i ] = term_list2


# In[9]:


test = [['hello', 'monther', ',', 'how', 'are', 'you', '?'], ['great', 'thanks']]

def del_stop_word(tokens):
    
    stop_words = nltk.corpus.stopwords.words( 'english' )

    # Take each tokenized wine review and remove stop words (i.e. only, own, don't, etc)
    for i in range( 0, len( tokens ) ):
        term_list2 = [ ]

        #Append terms that are not in stop word library into a new wine review vector
        for term in tokens[ i ]:
            if term not in stop_words:
                term_list2.append( term )

        #Replace the original wine review with the new wine review that doesn't contain stop words
        tokens[ i ] = term_list2
    return tokens

del_stop_word(test)


# In[11]:


### Remove puntuation from all of the term vectors

punc = re.compile( '[%s]' % re.escape( string.punctuation ) )
no_punc = []

for i in range(0, len( term_vec2 )):
    
    no_punc = []
    
    #Checks each term in a review and replaces any punctuation with nothing
    #EXAMPLE: Berry-Flavored ==> BerryFlavored
    for term in term_vec2[ i ]: 
        term = punc.sub( '', term )
        
        #Removes a few lingering characters not contained in the puncuation list
        if term != '' and term !='—' and term !='—' and term !='–':
            no_punc.append(term)
    
    #Replace the original term vector with the term vector without punctuation
    term_vec2[ i ] = no_punc 
    
len(term_vec2)


# In[12]:


test = [['hello', 'monther', ',', '?'], ['great', 'thanks']]

#This funtion expects a list of lists
def del_punc(tokens): 
    punc = re.compile( '[%s]' % re.escape( string.punctuation ) )
    no_punc = []

    for i in range(0, len( tokens )):

        no_punc = []

        #Checks each term in a review and replaces any punctuation with nothing
        #EXAMPLE: Berry-Flavored ==> BerryFlavored
        for term in tokens[ i ]: 
            term = punc.sub( '', term )

            #Removes a few lingering characters not contained in the puncuation list
            if term != '' and term !='—' and term !='—' and term !='–':
                no_punc.append(term)
                
        #Replace the original term vector with the term vector without punctuation
        tokens[ i ] = no_punc 
    return tokens

del_punc(test)


# In[13]:


### Performs Porter Stemming on the terms in the vector. 
### Replaces all terms in the term_vec2 with their porter stemmed equlalent

porter = nltk.stem.porter.PorterStemmer()

for i in range( 0, len( term_vec2 ) ):
    for j in range( 0, len( term_vec2[ i ] ) ):
        term_vec2[ i ][ j ] = porter.stem( term_vec2[ i ][ j ] )

len(term_vec2)


# In[14]:


test = [['hello', 'monthers'], ['great', 'thanks']]

def porter_stem(tokens): 
    
    porter = nltk.stem.porter.PorterStemmer()

    for i in range( 0, len( tokens ) ):
        for j in range( 0, len( tokens[ i ] ) ):
            tokens[ i ][ j ] = porter.stem( tokens[ i ][ j ] )
    
    return tokens

porter_stem(test)


# In[84]:


###  Convert term vectors into gensim dictionary to create the 
### Term Frequency–Inverse Document Frequency (TF-IDF) Matrix
### Reference https://www.csc2.ncsu.edu/faculty/healey/msa/text/ for more information regarding
### creation of the TF-IDF

# This creates a dictionary of terms each with a unique term ID
# So "Many" may be given an ID of 2
# The ID's will be used in the next step

dict = gensim.corpora.Dictionary( term_vec2 )


corp = [ ]
for i in range( 0, len( term_vec2 ) ):
    
    # For each term vector this will match each term with its term ID and give the count of that term
    # EXAMPLE: "to" has ID=4 and "chicken" has ID=8. The sentence "chicken to chicken to to" becomes
    # [(4, 3), (8, 1)]
    corp.append( dict.doc2bow( term_vec2[ i ] ) )

#  Create TFIDF vectors based on term vectors bag-of-word corpora
# This takes the corpa made in the previous step and calculates the 
# TFIDF for each term in each review
tfidf_model = gensim.models.TfidfModel( corp )

tfidf = [ ]
for i in range( 0, len( corp ) ):
    tfidf.append( tfidf_model[ corp[ i ] ] )
    
###  Create pairwise document similarity index
n = len( dict )

#Index will contain all of the document to document similarities
#Similarity is calculated by calculating cosine similarity
index = gensim.similarities.SparseMatrixSimilarity( tfidf_model[ corp ], num_features = n )

sim = index[ tfidf_model[ corp[ i ] ] ]


# In[58]:


#term_vec2=term_vec2[:-1]


# In[102]:




def get_recommended(tokens, winedata=None, min_points=None, max_price=None, num_recs = 5): 
    ###  Convert term vectors into gensim dictionary to create the 
    ### Term Frequency–Inverse Document Frequency (TF-IDF) Matrix
    ### Reference https://www.csc2.ncsu.edu/faculty/healey/msa/text/ for more information regarding
    ### creation of the TF-IDF

    # This creates a dictionary of terms each with a unique term ID
    # So "Many" may be given an ID of 2
    # The ID's will be used in the next step
    
    ### Process the user input:
    new_review = [[input(" What type of wine are you looking for? ")]]
    
    new_review = tokenize(new_review)
    
    new_review = del_punc(new_review)
    
    new_review = del_stop_word(new_review)
    
    new_review = del_punc(new_review)
    
    new_review = porter_stem(new_review)
    
    
    ### Filter data based on points
    #Get index values of wines meeting the input criteria
    to_keep = wine_data[(wine_data['points'] >= min_points) & (wine_data['price']<= max_price)].index.values

    input_tokens = []

    count = 0
    
    # This will output the tokenized term vectors that meet user input criteria into a 
    # subsetted term vector that similarity analysis will be perfromed on
    
    for i in tokens:
        if count in to_keep:
            input_tokens.append(i)
        count+=1
    
    wine_df = wine_data[wine_data.index.isin(to_keep)]
    
    input_tokens.append(new_review[0])

    dict = gensim.corpora.Dictionary( input_tokens )


    corp = [ ]
    for i in range( 0, len( input_tokens ) ):

        # For each term vector this will match each term with its term ID and give the count of that term
        # EXAMPLE: "to" has ID=4 and "chicken" has ID=8. The sentence "chicken to chicken to to" becomes
        # [(4, 3), (8, 1)]
        corp.append( dict.doc2bow( input_tokens[ i ] ) )

    #  Create TFIDF vectors based on term vectors bag-of-word corpora
    # This takes the corpa made in the previous step and calculates the 
    # TFIDF for each term in each review
    tfidf_model = gensim.models.TfidfModel( corp )

    tfidf = [ ]
    for i in range( 0, len( corp ) ):
        tfidf.append( tfidf_model[ corp[ i ] ] )

    ###  Create pairwise document similarity index
    n = len( dict )

    #Index will contain all of the document to document similarities
    #Similarity is calculated by calculating cosine similarity
    index = gensim.similarities.SparseMatrixSimilarity( tfidf_model[ corp ], num_features = n )
    
    #Get the similarity values from the user input vector
    sims = index[ tfidf_model[ corp[ -1 ] ] ]
    
    #merge similaries with original dataset to display recommended wines
    wine_df['sims'] = sims[:-1]
    
    return wine_df.sort_values(by=['sims'], ascending=False).head(num_recs)


# In[105]:


get_recommended(term_vec2, min_points=90, max_price=30, num_recs = 5)


# In[106]:


len(term_vec2)


# In[101]:


to_keep = wine_data[(wine_data['points'] >=93) & (wine_data['price']<= 15)].index.values

print(to_keep)

wine_data[wine_data.index.isin(to_keep)].drop(['variety', 'winery', 'designation'], axis=1)


# In[92]:


type(to_keep)


# In[47]:


sub_data = wine_data.copy()

sub_data['sims'] = sims[:-1]

sub_data.sort_values(by=['sims'], ascending=False).head(6)


# In[41]:


test = gensim.models.TfidfModel( corp[1:4] )
print(test[ corp[1] ])
print(test[ corp[2] ])
print(corp[1])
print(corp[2])
print(term_vec2[2])
dict[6]


# In[11]:


#  Create pairwise document similarity index
n = len( dict )

#Index will contain all of the document to document similarities
index = gensim.similarities.SparseMatrixSimilarity( tfidf_model[ corp ], num_features = n )

'''

#  Print TFIDF vectors and pairwise similarity per document

for i in range( 0, len( tfidf ) ):
    s = 'Doc ' + str( i + 1 ) + ' TFIDF:'

    for j in range( 0, len( tfidf[ i ] ) ):
        s = s + ' (' + dict.get( tfidf[ i ][ j ][ 0 ] ) + ','
        s = s + ( '%.3f' % tfidf[ i ][ j ][ 1 ] ) + ')'

    print s

for i in range( 0, len( corp ) ):
    print 'Doc', ( i + 1 ), 'sim: [ ',

    sim = index[ tfidf_model[ corp[ i ] ] ]
    for j in range( 0, len( sim ) ):
        print '%.3f ' % sim[ j ],

    print ']
    
'''


# In[ ]:








# In[8]:


'''

### Get frequently used words from the word vectors

cnt_list = [[],[]]
for set_i in corp:
    for x in set_i: 
        cnt_list[0].append(dict[x[0]])
        cnt_list[1].append(x[1])

word_counts = pd.DataFrame([cnt_list]).transpose()

get_word_cnts = word_df_s.groupby([0]).sum()

sorted_cnts = get_word_cnts.sort_values(by=[1], ascending=False)

popular_words = sorted_cnts[sorted_cnts[1]>10000]

popular_words['ratio'] = (popular_words[1] / len(word_df))*100

popular_words.head()

'''

