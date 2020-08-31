#Author: Mobashir Sadat
#msadat3
#668038298

import re
from nltk.stem import PorterStemmer
import numpy as np

#Class to represent every document
class Document():
    def __init__(self, location, gold_label_location, Stopwords):
        super(Document, self).__init__()
        self.location = location
        self.gold_label_location = gold_label_location
        self.text = ''
        self.preprocessed = []
        self.stopwords = Stopwords
        with open(location, 'rb', ) as fp:
            self.text = fp.read()
            self.text = self.text.decode("utf-8")
        self.vocab= None
        self.adjacency_matrix = None
        self.gold_labels = []


    def Preprocess(self):#preprocessing
        stemmer = PorterStemmer()
        stopwords = self.stopwords

        self.text = self.text.replace('\n',' ')
        self.text = self.text.lower()
        self.text = self.text.split(' ')

        for i in range(len(self.text)):
            #NN, NNS, NNP, NNPS, JJ
            if ('_nn' in self.text[i]) or ('_nns' in self.text[i]) or ('_nnp' in self.text[i]) or ('_nnps' in self.text[i]) or ('_jj' in self.text[i]):##only keeping the jectives and nouns
                temp = re.sub(r'_.*$', '', self.text[i])#regex for getting rid of the POS terms
                temp = stemmer.stem(temp)
                self.text[i] = temp
                if temp not in stopwords:
                    self.preprocessed.append(temp)
            else:
                temp = re.sub(r'_.*$', '', self.text[i])
                self.text[i] = temp
        self.vocab = list(set(self.preprocessed))

    def word_to_idx(self, word): #returns the index of a word in the vocabulary
        return self.vocab.index(word)

    def Create_word_graph(self,window):##creating the adjacency matrix
        adjacency_matrix = np.zeros((len(self.vocab),len(self.vocab)),dtype=int)
        window = window-1
        for word in self.vocab:
            occurances = [i for i, x in enumerate(self.text) if x == word]
            for i in occurances:
                window_start = i-window
                window_end = i+window
                if window_start <0:
                    window_start = 0
                if window_end > len(self.text)-1:
                    window_end = len(self.text)-1
                for j in range(window_start,window_end+1):
                    if (self.text[j]!=word) and (self.text[j] in self.vocab): #adding an edge only if the words are still adjacent after preprocessing(are members in the vocabulary)
                        adjacency_matrix[self.word_to_idx(word), self.word_to_idx(self.text[j])]+=1

        self.adjacency_matrix =  adjacency_matrix



    def get_adjacent_indices(self,idx):##return the indices adjacent to a word in the adjacency matrix
        adjacent_indices = np.nonzero(self.adjacency_matrix[idx])
        return adjacent_indices[0]

    def update_scores(self,scores, alpha):#one ieteration of page rank
        new_scores = {}
        for word in self.vocab:
            i = self.word_to_idx(word)
            adjacent_indices_i = self.get_adjacent_indices(i)
            sum = 0
            for j in adjacent_indices_i:

                numerator = self.adjacency_matrix[j,i]
                denominator = 0
                adjacent_indices_j = self.get_adjacent_indices(j)
                for k in adjacent_indices_j:
                    denominator+=self.adjacency_matrix[j,k]
                sum+= (numerator/denominator) * scores[self.vocab[j]]
            new_scores[word] = (alpha * sum) + ((1-alpha)*(1/len(self.vocab)))
        return new_scores

    def get_page_rank_scores(self, alpha,num_iterations):#page rank for every word
        scores = {}
        for word in self.vocab:
            scores[word] = 1/len(self.vocab)
        iter = 0
        converged = False
        while (iter < num_iterations) and (converged == False):
            new_scores = self.update_scores(scores,alpha)
            if scores == new_scores:
                converged = True
            scores = new_scores
            iter+=1
        return scores

    def get_phrases(self): #create phrases
        unigrams = [[x] for x in self.vocab]
        bigrams = []
        trigrams = []
        for i in range(len(self.text)-1):
            if (self.text[i] in self.vocab) and (self.text[i+1] in self.vocab):
                bigrams.append(self.text[i:i+2])

        for i in range(len(self.text)-2):
            if (self.text[i] in self.vocab) and (self.text[i+1] in self.vocab) and (self.text[i+2] in self.vocab):
                bigrams.append(self.text[i:i+3])

        phrases = unigrams + bigrams + trigrams

        return phrases

    def get_top_k_phrases(self, k,alpha, num_iterations): # get phrase scores and rank
        phrases = self.get_phrases()
        word_scores = self.get_page_rank_scores(alpha, num_iterations)
        phrase_scores = {}
        for phrase in phrases:
            phrase_scores[' '.join(phrase)] = 0
            for word in phrase:
                phrase_scores[' '.join(phrase)]+= word_scores[word]

        phrases_sorted = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_phrases = [x[0] for x in phrases_sorted[0:k]]

        return top_k_phrases


    def get_gold_labels(self): #function to read gold label files and preprcocess gold labels
        stemmer = PorterStemmer()
        with open(self.gold_label_location, 'rb', ) as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.decode("utf-8")
                line = line.replace('\n','')
                words = line.split(' ')
                for i in range(len(words)):
                    words[i] = stemmer.stem(words[i])
                temp = ' '.join(words)
                self.gold_labels.append(temp)

    def get_reciprocal_rank(self, k,alpha, num_iterations):#returns reciprocal rank of one document
        candidates = self.get_top_k_phrases(k,alpha, num_iterations)
        self.get_gold_labels()
        for i in range(len(candidates)):
            if candidates[i] in self.gold_labels:
                return 1/(i+1)
        return 0























