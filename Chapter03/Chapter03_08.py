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
