def calculate_perplexity(sent, prob_matrix):
    ppl = 1
    for i in range(1,len(sent)-2):
        w1, w2 = sent[i], sent[i+1]
        #ppl *= probability_matrix[(w1,w2)]
        w1 = (w1,)
        ppl *= prob_matrix[w2][w1]
    return pow(ppl, -1/(len(sent)-2))
