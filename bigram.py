from brown_corpus import get_sentences_with_word2index_vocab_limit
import numpy as np
import sys


class Bigram:
    '''
    Bigram model created using Brown corpus

    use 
        get_score_for_sentence(sentence) to get score

    '''
    def __init__(self):
        self.V = 0
        self.start_index = 0
        self.end_index = 0
        self.index2word = dict()
        self.bigram_probs = np.array([])
        self.word2index = dict()
        self.main()
        
    @property
    def V(self):
        return self._V
    
    @V.setter
    def V(self, V):
        self._V = V
        
    @property
    def start_index(self):
        return self._start_index
    
    @start_index.setter
    def start_index(self, start_index):
        self._start_index = start_index
    
    @property
    def end_index(self):
        return self._end_index
    
    @end_index.setter
    def end_index(self, end_index):
        self._end_index = end_index
    
    @property
    def bigram_probs(self):
        return self._bigram_probs
    
    @bigram_probs.setter
    def bigram_probs(self, bigram_probs):
        self._bigram_probs = bigram_probs
        
    @property
    def word2index(self):
        return self._word2index
    
    @word2index.setter
    def word2index(self, word2index):
        self._word2index = word2index
        
    @property
    def index2word(self):
        return self._index2word
    
    @index2word.setter
    def index2word(self, index2word):
        self._index2word = index2word
        
    
    def __get_bigram_probability(self, sentences, smoothing=1):
        ##get probablity of sentences
        ##create a matrix of vocabularies
        bigram_probabilty = np.ones((self.V,self.V)) * smoothing
        ## row side(vertical) will be last word (previous word), columns side(horizontal) will be next word
        ## (last_word, next_word) == (row, column) == (previous word, next word)

        ##in matrix start and end will be in columns 0,1 and row 0,1
        ##so if iteration is starting we will update 0,0 as +=1
        ##if it is ending we will update 

        for sentence in sentences:
            #get a single line of word index
            for i in range(len(sentence)):

                #starting word
                if i == 0:
                    bigram_probabilty[self.start_index, i] +=1

                #others
                else:
                    bigram_probabilty[sentence[i-1], i]+=1
            if i == len(sentence) -1:
                #if it is last word of sentence, there wont be any next word, so take end index position and update
                bigram_probabilty[sentence[i], self.end_index] +=1

        ## sum will sum all the values along the row size (becoz axis = 1) and create a new array
        ##new array will be a 1d array, will sum of all rows, list of size 1000 if sumed array has 1000 rows 
        ## to keep the dimension ie., keep the shape (1000,1) use keepdims = True
        ## then divide it to the bigram to get prob
        ## prob = count of samples / population

        bigram_probabilty /= bigram_probabilty.sum(axis = 1, keepdims = True)

        return bigram_probabilty

        
    
    def __get_words(self, sentence):
        '''
        return sentence in words when you pass sentence in list of index form
        '''
        return ' '.join([self.index2word[word_idx] for word_idx in sentence])
    
    def get_score(self, sentence):
        '''
        get log probabilty of sentence
        when we do normal prob ie p(a) * p(b) * .... p(inf), it will increase decimal
        since prob is between zero and one, value will become very low to be stored by memory
        0.1 * 0.1 = 0.01, if we multiplied it further it will become zero
        since we know log is always between 0 and 1 we can use that
        in logs multiplication is addition, so it will help
        '''
        score = 0
        for i in range(len(sentence)):

            #start of word
            if i ==0:
                score += np.log(self.bigram_probs[self.start_index, i])
            else:
                #middle word
                score += np.log(self.bigram_probs[i-1, i])

            if i == len(sentence)-1:
                ##last word
                score += np.log(self.bigram_probs[i, self.end_index])

        #normalize the score. +1 is to accomodate start and end index
        return score/ (len(sentence)+1)
    
    
    def main(self):
        sentences, self.word2index = get_sentences_with_word2index_vocab_limit(vocab_size = 1000)
        self.V = len(self.word2index)
        self.start_index = self.word2index["START"]
        self.end_index = self.word2index["END"]
        self.bigram_probs = self.__get_bigram_probability(sentences,smoothing=0.1)

        ##since we have word to index, create index to word dict
        self.index2word = {v:k for k,v in self.word2index.items()}
        
        ##create fake sentence list
        sample_probs = np.ones(self.V)
        sample_probs[self.start_index] = 0
        sample_probs[self.end_index] = 0
        sample_probs /= sample_probs.sum()

        real_idx = np.random.choice(len(sentences))
        real = sentences[real_idx]

        # fake sentence
        fake = np.random.choice(self.V, size=len(real), p=sample_probs)
        print("REAL:", self.__get_words(real), "SCORE:", self.get_score(real))
        
    def get_score_for_sentence(self, sentence):
        sentence = sentence.lower().split()

        # check that all tokens exist in word2idx (otherwise, we can't get score)
        bad_sentence = False
        for word in sentence:
            if word not in self.word2index:
                bad_sentence = True

        if bad_sentence:
            return "Sorry, you entered words that are not in the vocabulary"
        else:
            # convert sentence into list of indexes
            sentene_in_index = [self.word2index[token] for token in sentence]
            score = self.get_score(sentene_in_index)
            print("SCORE:", score)
            return score
        

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Enter some sentence')
    else:
        sentence = sys.argv[1]
        bigram = Bigram() 
        score = bigram.get_score_for_sentence(sentence)       