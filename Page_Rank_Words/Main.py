#Author: Mobashir Sadat
#msadat3
#668038298

from Document import *
from nltk.stem import PorterStemmer
import os
import sys
import os.path as p

print(sys.argv)

stopwordsLocation = sys.argv[1]
abstractLocation = sys.argv[2]+'/'
goldLabelLocation = sys.argv[3]+'/'
window = int(sys.argv[4])

def getStopWordsList(fileLocation):  ##get the stop words from a file
    stopwords = []
    stemmer = PorterStemmer()
    with open(fileLocation, 'rb') as file:
        for line in file:
            word = line.decode("utf-8")
            word = word.strip('\n')
            word = stemmer.stem(word)
            stopwords.append(word)
    return stopwords

stopwords = getStopWordsList(stopwordsLocation)

#gets MRR by extracting keyphrases from all documents given the locations, value of k, alpha, highet number of iterations and window size
def calculate_MRR(document_directory_location, gold_label_directory_location, stopwords, k,alpha, num_iterations, window):
    Reciprocal_ranks = []
    for root, dirs, files in os.walk(document_directory_location):
        for file in files:
            if p.exists(gold_label_directory_location+file):
                doc = Document(document_directory_location+file, gold_label_directory_location+file, stopwords)
                doc.Preprocess()
                doc.Create_word_graph(window)
                rec = doc.get_reciprocal_rank(k,alpha, num_iterations)
                #print(file,rec)
                Reciprocal_ranks.append(rec)
    mrr =  sum(Reciprocal_ranks) / len(Reciprocal_ranks)
    print('Window =', window, "K =",k,'MRR =',mrr)

Ks = [1,2,3,4,5,6,7,8,9,10]

for k in Ks:
    calculate_MRR(abstractLocation, goldLabelLocation, stopwords, k, alpha=0.85,
                  num_iterations=10, window=window)

#python Main.py "D:/CS 582/HW4/Page_Rank_Words/stopwords.txt" "D:/CS 582/HW4/www/abstracts" "D:/CS 582/HW4/www/gold/" 2

'''doc = Document(abstractLocation + '10348116', goldLabelLocation + '10348116', stopwords)
doc.Preprocess()
doc.Create_word_graph(window)
top = doc.get_top_k_phrases(k=10, alpha=0.85, num_iterations=10)
print('text')
print(doc.text)
print('preprocessed')
print(doc.preprocessed)
print('gold:')
doc.get_gold_labels()
print(doc.gold_labels)

print('candidates')
print(top)'''

