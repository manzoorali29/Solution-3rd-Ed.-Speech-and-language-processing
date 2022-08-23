# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from collections import defaultdict
import re
import random
## Preprocessing

def sentence_corpus(file, start_token = "<s>", end_token = "</s>"):
    """
  
    A simple function which took a file which contains sentences line by line and returns a list of sentences includin
    start and end token.
    
    Args:
        file: line by line sentences.
        start_token: default value = <s>
        end_token: default value = </s>
    
    Returns:
        sentences: list of all sentences with start and end tokens
        tokens : combine all the sentences to a single list of tokens
    """
    
    sentences = []
    
    tokens = []
    temp = []
    file = open(file, "r")
    for line in file:
        
        line = line.lower()
        line = re.sub(r'[^a-zA-Z0-9/s]',' ',line)
        line = start_token+" "+ line+" " + end_token
        temp = [token for token in line.split(" ") if token != ""]
        sentences.append(temp)
        for t in temp:
            tokens.append(t)
        
    return sentences, tokens


#print(sentence_corpus('sample.txt'))

        
def single_pass_ngram_count_matrix(corpus,n):
    """
    Creates the ngram count matrix from the input corpus in a single pass through the corpus.
    
    Args:
        corpus: Pre-processed and tokenized corpus. 
        n: represents the ngram i.e. bi, tri or any
    
    Returns:
        ngrams: list of all ngrams prefixes, row index
        vocabulary: list of all found words, the column index
        count_matrix: pandas dataframe with ngrams prefixes as rows, 
                      vocabulary words as columns 
                      and the counts of the ngram/word combinations (i.e. trigrams) as values
    """
    ngrams = []
    vocabulary = []
    count_matrix_dict = defaultdict(dict)
    
    # go through the corpus once with a sliding window
    for i in range(len(corpus) - n + 1):
        # the sliding window starts at position i and contains 3 words
        n_plusone_gram = tuple(corpus[i : i + n])
        
        gram = n_plusone_gram[0 : -1]
        if not gram in ngrams:
            ngrams.append(gram)        
        
        last_word = n_plusone_gram[-1]
        if not last_word in vocabulary:
            vocabulary.append(last_word)
        
        if (gram,last_word) not in count_matrix_dict:
            count_matrix_dict[gram,last_word] = 0
            
        count_matrix_dict[gram,last_word] += 1
    
    # convert the count_matrix to np.array to fill in the blanks
    count_matrix = np.zeros((len(ngrams), len(vocabulary)))
    for trigram_key, trigam_count in count_matrix_dict.items():
        count_matrix[ngrams.index(trigram_key[0]), \
                     vocabulary.index(trigram_key[1])]\
        = trigam_count
    
    # np.array to pandas dataframe conversion
    count_matrix = pd.DataFrame(count_matrix, index=ngrams, columns=vocabulary)
    return ngrams, vocabulary, count_matrix

def laplace_addOne_smoothing(count_matrix,V):
    """
    It smooth a count matrix by applying add one smoothing without dividing on the length(only add one matrix)
    
    Args:
        count_matrix: pandas dataframe with ngrams prefixes as rows, 
                      vocabulary words as columns 
                      and the counts of the ngram/word combinations (i.e. trigrams) as values
        V: length of vocabulary (Integer)
    
    Returns:
        count_matrix_smooth: a smooth matrix
        row_sum: sum of all rows of smooth matrix
        
    """
    count_matrix_smooth = count_matrix + 1
    row_sum = count_matrix_smooth.sum(axis=1)
    row_sum = row_sum  + V
    
    return count_matrix_smooth,row_sum
    
    
    
def probibility_matrix(count_matrix,V,smoothing=None):
    """
    This function calculates a probibility matrix of ngrams.
    
    Args:
        count_matrix: pandas dataframe with ngrams prefixes as rows, 
                      vocabulary words as columns 
                      and the counts of the ngram/word combinations (i.e. trigrams) as values
        V: length of vocabulary (Integer)
        smoothing: if None there will be no smoothing applies otherwise laplace smoothing
    
    Returns:
        prob_matrix: pandas dataframe with ngrams prefixes as rows, 
                      vocabulary words as columns 
                      and the probibility of the ngram/word combinations (i.e. trigrams) as values
        
        
    """
    if smoothing == None:    
        row_sum = count_matrix.sum(axis=1)
        prob_matrix = count_matrix.div(row_sum,axis=0)
    else:
        c_matrix,row_sum = laplace_addOne_smoothing(count_matrix,V)
        
        prob_matrix = c_matrix.div(row_sum,axis=0)
    return prob_matrix

def find_probibility(prob_matrix,ngram):
    """
        This function find the probibility of an ngram.
        
        Args:
            prob_matrix: pandas dataframe with ngrams prefixes as rows, 
                          vocabulary words as columns 
                          and the probibility of the ngram/word combinations (i.e. trigrams) as values
            ngram: an ngram
           
        
        Returns:
            ngram_probability: probibility value against an ngram and the word
            
        
    """
    n_m_one_gram = ngram[:-1]
    print(f'n minus one gram: {n_m_one_gram}')
    word = ngram[-1]
    print(f'word: {word}')
    ngram_probability = prob_matrix[word][n_m_one_gram]
    print(f'ngram_probability: {ngram_probability}')
    return ngram_probability


def estimate_probabilities(vocabulary):
    """
    Estimate the probabilities of next words using the n-gram counts with k-smoothing
    
    Args:
        previous_n_gram: A sequence of words of length n
        n_gram_counts: Dictionary of counts of (n+1)-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary: List of words
        k: positive constant, smoothing parameter
    
    Returns:
        A dictionary mapping from next words to the probability.
    """
    
    # convert list to tuple to use it as a dictionary key
   
    
    # add <e> <unk> to the vocabulary
    # <s> is not needed since it should not appear as the next word
    vocabulary = vocabulary + ["<e>", "<unk>"]
    vocabulary_size = len(vocabulary)
    
    probabilities = {}
    for word in vocabulary:
        probability = find_probibility()
        probabilities[word] = probability

    return probabilities

def rand_sentences(probability_matrix,vocab,num_sent):
    for i in range(num_sent):
        
   
    
    # starting words
        text = [('<s>',)]
        sentence_finished = False
     
        while not sentence_finished:
      # select a random probability threshold  
            r = random.random()
           
            accumulator = .0
           
           
            for word in vocab:
                accumulator += probability_matrix[word][text[-1]]
              # select words that are above the probability threshold
                if accumulator >= r:
                    text.append((word,))
                    break
        
            if text[-1] == ('</s>',):
                sentence_finished = True
                print (' '.join([t[0] for t in text if t]))
     
def calculate_perplexity(sent, prob_matrix):
    ppl = 1
    for i in range(1,len(sent)-2):
        w1, w2 = sent[i], sent[i+1]
        #ppl *= probability_matrix[(w1,w2)]
        w1 = (w1,)
        ppl *= prob_matrix[w2][w1]
    return pow(ppl, -1/(len(sent)-2))
_, corpus = sentence_corpus('sample.txt')
bigrams, vocabulary, count_matrix = single_pass_ngram_count_matrix(corpus,2)
p_matrix = probibility_matrix(count_matrix,len(vocabulary),smoothing='yes')
ngram = ('einstein','born')
find_probibility(p_matrix,ngram)
print(rand_sentences(p_matrix, vocabulary,3))
print(calculate_perplexity([ '1903', 'his', 'kept', 'diploma', 'polytechnic', 'for', 'citizen', '</s>'],p_matrix))

print(calculate_perplexity(['<s>','in', '1905', 'he', 'was', 'awarded', 'a', 'phd', 'by', 'the', 'university', 'of', 'zurich','</s>'], p_matrix))


