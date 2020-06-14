from nltk import corpus
from operator import itemgetter

KEEP_WORDS = set([
  'king', 'man', 'queen', 'woman',
  'italy', 'rome', 'france', 'paris',
  'london', 'britain', 'england',
])

def get_sentences():
    '''
    return sentences from nltk brown corupus
    totaly 57340 list of sentences [[sent1], [sent2]...]
    sent1 = [word1, word2, ...]
    each sentence has list of words
    
    '''
    return corpus.brown.sents()

def get_senteneces_with_word2index():
    #get sentence
    sentences = get_sentences()
    word2idx = {'START' : 0, 'END' : 1}
    index = 2
    indexed_sentences = [] #all sentence
    for sentence in sentences:
        indexed_sentence = [] #current sentence in iteration
        for word in sentence:
            word = word.lower()
            if word not in word2idx:
                word2idx[word] = index
                index +=1
            indexed_sentence.append(word2idx[word])

        indexed_sentences.append(indexed_sentence)
        
    print('total vocabulary :', index)
        
    return indexed_sentences, word2idx    

indexed_sentences, word2idx = get_senteneces_with_word2index()

def get_sentences_with_word2index_vocab_limit(vocab_size = 2000, keepwords= KEEP_WORDS):
    #get sentence
    sentences = get_sentences()
    word2idx = {'START' : 0, 'END' : 1}
    index = 2
    indexed_sentences = [] #all sentence
    words_indexed = ['START','END']
    
    #we are going to sort based on count. so keep it infinity to avoid deleting it
    word_index_count = {
        0 : float('inf'),
        1 : float('inf')
    }
    
    for sentence in sentences:
        indexed_sentence =[] #current sentence in iteration
        for word in sentence:
            word = word.lower()
            if word not in word2idx:
                word2idx[word] = index
                index +=1
                words_indexed.append(word)
            index_of_word = word2idx[word]
            
            #for sorting based on count purpose
            word_index_count[index_of_word] = word_index_count.get(index_of_word, 0) + 1
            indexed_sentence.append(index_of_word)
        indexed_sentences.append(indexed_sentence)
    ##now reduce the size of vocab
    ##mark keep words as infinity to avoid removal
    
    for keep_word in keepwords:
        word_index_count[word2idx[keep_word]] = float('inf')
        
    
    sorted_counts = sorted(word_index_count.items(), key = itemgetter(1), reverse=True)
    
    word2idx_small = {}
    word2idx_old2new_map = {}
    
    new_index = 0
    for old_index, _ in sorted_counts[:vocab_size]:
        word = words_indexed[old_index]
        word2idx_small[word] = new_index
        word2idx_old2new_map[old_index] = new_index
        new_index += 1
        
    word2idx_small['UNKNOWN'] = new_index
    unknown_index= new_index
    sentences_small = []
        
    for sentence in indexed_sentences:
        if len(sentence) > 1:
            small_sentence = []
            for word_index in sentence:
                word = word2idx_old2new_map.get(word_index, unknown_index)
                small_sentence.append(word)
            sentences_small.append(small_sentence)
            
    return sentences_small, word2idx_small

sent2, word_index2= get_sentences_with_word2index_vocab_limit()