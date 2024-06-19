import re
from nltk.stem import PorterStemmer 
import math
from tqdm import tqdm
import os
import pickle
class QueryParsers:

    def __init__(self, query):
        # self.filename = file
        self.query= self.get_queries(query)

    def get_queries(self, query):
        # q = open(self.filename,'r').read().lower()
        q = query.lower()
        #subsitute all non-word characters with whitespace
        pattern = re.compile('\W+')
        q = pattern.sub(' ', q)
        # split text into words (tokenized list for a document)
        q = q.split()
        # stemming words
        stemmer = PorterStemmer()
        q = [stemmer.stem(w) for w in q ]
        return q

def concat_docs(corpus, sep = " "):
    if type(corpus) is dict:
        sentences = [(corpus["title"][i] + sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
    else:
        if type(corpus) is list:
            if type(corpus[0]) is dict:
                sentences = [(doc["title"] + sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
            else:
                sentences = corpus
    return sentences

class BuildIndex:
	
    b = 0.75
    k = 1.2


    def __init__(self, sample_hash, doc_ls=None):
        self.tf = {}
        self.df = {}
        # self.filenames = files
        self.file_to_terms = self.process_files(sample_hash, doc_ls=doc_ls)
        self.regdex = self.regular_index(self.file_to_terms)
        self.invertedIndex = self.inverted_index()
        self.dltable = self.docLtable()
        self.dl = self.docLen()
        self.avgdl = self.avgdocl()
        self.N = self.doc_n()
        self.idf = self.inverse_df()

    def retrieval(self, queries):
        # q = QueryParsers('queries.txt')
        # query = q.query
        total_scores_mappings = dict()
        for qid in tqdm(range(len(queries))):
            query = queries[qid]
            q = QueryParsers(query)
            curr_scores  = self.BM25scores(q.query)
            total_scores_mappings[str(qid+1)] = curr_scores
            # self.rankedDocs = self.ranked_docs()
        return total_scores_mappings

    def process_files(self, sample_hash, doc_ls=None):
        # '''
        # input: filenames
        # output: a dictionary keyed by filename, and with values of its term list
        # '''
        
        index_file_name = "output/bm25_index_" + sample_hash + ".pkl"
        if os.path.exists(index_file_name):
            with open(index_file_name, "rb") as f:
                return pickle.load(f)
        
        file_to_terms = []

        doc_ls = concat_docs(doc_ls)

        # if doc_ls is not None:
        #     key_ls = list(doc_ls.keys())
        # else:
        #     key_ls = self.filenames

        for file in tqdm(range(len(doc_ls)), desc='Processing documents'):
            #read the whole text of a file into a single string with lowercase
            # if doc_ls is None:
            #     file_to_terms[file] = open(file,'r').read().lower()
            # else:
            process_str = doc_ls[file].lower() 
            #subsitute all non-word characters with whitespace
            pattern = re.compile('\W+')
            process_str = pattern.sub(' ', process_str)
            # split text into words (tokenized list for a document)
            process_str = process_str.split()
            # stemming words
            stemmer = PorterStemmer()
            file_to_terms.append([stemmer.stem(w) for w in process_str])
        
        pickle.dump(file_to_terms, open(index_file_name, "wb"))
        
        return file_to_terms
        # return file_to_terms

    def doc_n(self):
        '''
        return the number of docs in the collection
        '''
        return len(self.file_to_terms)


    def index_one_file(self, termlist):
        '''
        input: termlist of one document.
        map words to their position for one document
        output: a dictionary with word as key, position as value.
        '''
        fileIndex = {}
        for index,word in enumerate(termlist):
            if word in fileIndex.keys():
                fileIndex[word].append(index)
            else:
                fileIndex[word] = [index]

        return fileIndex

    def regular_index(self,termlists):
        '''
        input: output of process_files(filenames)
        output: a dictionary. key: filename, value: a dictionary with word as key, position as value  
        '''
        regdex = []

        for filename in range(len(termlists)):
            regdex.append(self.index_one_file(termlists[filename]))

        return regdex


    def inverted_index(self):
        '''
        input: output of make_indexes function.
        output: dictionary. key: word, value: a dictionary keyed by filename with values of term position for that file.
        '''
        total_index = {}
        regdex = self.regdex

        for filename in range(len(regdex)):
            
            self.tf[filename] = {}

            for word in regdex[filename].keys():
                # tf dict key: filename, value: dict key is word, value is count
                self.tf[filename][word] = len(regdex[filename][word])
                
                if word in self.df.keys():
                    # df dict key: word, value: counts of doc containing that word
                    self.df[word] += 1
                else:
                    self.df[word] = 1

                if word in total_index.keys():
                    if filename in total_index[word].keys():
                        total_index[word][filename].extend(regdex[filename][word])
                    else:
                        total_index[word][filename] = regdex[filename][word]
                else:
                    total_index[word] = {filename: regdex[filename][word]}

        return total_index

    def docLtable(self):
        '''
        output: dict, key:word, value:dict(key: number of docs contaiing that word, value:total_freq)
        '''
        dltable = {}
        for w in tqdm(self.invertedIndex.keys(), desc='Building doc length table'):	
            total_freq = 0
            for file in self.invertedIndex[w].keys():
                total_freq += len(self.invertedIndex[w][file])
            
            dltable[w] = {len(self.invertedIndex[w].keys()):total_freq}
        
        return dltable


    def docLen(self):
        '''
        return a dict, key: filename, value: document length
        '''
        dl = []
        for file in range(len(self.file_to_terms)):
            dl.append(len(self.file_to_terms[file]))
        return dl

    def avgdocl(self):
        sum = 0
        for file in range(len(self.dl)):
            sum += self.dl[file]
        avgdl = sum/len(self.dl)
        return avgdl


    def inverse_df(self):
        '''
        output: inverse doc freq with key:word, value: idf
        '''
        idf = {}
        for w in tqdm(self.df.keys(), desc='Building inverse doc freq'):
            # idf[w] = math.log((self.N - self.df[w] + 0.5)/(self.df[w] + 0.5))
            idf[w] = math.log((self.N +1 )/self.df[w])
        return idf


    def get_score (self,filename,qlist):
        '''
        filename: filename
        qlist: termlist of the query 
        output: the score for one document
        '''
        score = 0
        for w in self.file_to_terms[filename]:
            if w not in qlist:
                continue
            wc = len(self.invertedIndex[w][filename])
            score += self.idf[w] * ((wc)* (self.k+1)) / (wc + self.k * 
                                                            (1 - self.b + self.b * self.dl[filename] / self.avgdl))
        return score


    def BM25scores(self,qlist):
        '''
        output: a dictionary with filename as key, score as value
        '''
        total_score = {}
        score_sum = 0
        for doc in range(len(self.file_to_terms)):
            curr_score = self.get_score(doc,qlist)
            total_score[str(doc + 1)] = curr_score
            score_sum += curr_score
        total_score = {k: v/score_sum for k, v in total_score.items()}
        return total_score


    def ranked_docs(self):
        ranked_docs = sorted(self.total_score.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs