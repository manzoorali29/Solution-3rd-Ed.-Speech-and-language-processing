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
