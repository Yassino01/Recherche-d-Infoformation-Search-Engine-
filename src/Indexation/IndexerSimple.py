from utils.TextRepresenter import PorterStemmer
import math
import json

class IndexerSimple:

    ############# STATICS ATTRIBUTES #####################
    stem = PorterStemmer()
    save_folder = "../data/indexes/"
    ############# CLASS FUNCTIONS    #####################
    def __init__(self):
        pass
    
    def getStrDoc(self,doc):
        '''Return str from doc id in source file '''
        return self.corpus[doc].data.get('T','')+' '+self.corpus[doc].getText()

    def indexation(self,corpus,corpus_name,write_to_file = False):
        '''Compute,store and returns (index,invertedIdex)'''
        self.corpus = corpus
        self.index = IndexerSimple.indexFromCorpus(corpus)
        self.inverted_index = IndexerSimple.invertIndex(self.index)
            ### ecriture fichier 
        if write_to_file :
            IndexerSimple.write_dict(corpus_name+"_index",self.index)
            IndexerSimple.write_dict(corpus_name+"_inverted",self.inverted_index)
        
        return (self.index,self.inverted_index)
    
    def getTfsForDoc(self,doc):
        '''Return the (stem-TF) representation from doc id'''
        return self.index[doc]
    
    def getTfsIDFsForDoc(self,doc):
        '''Returns the (stem-TFIDF) representation from doc id'''
        TFs = self.getTfsForDoc(doc)
        return {t : tf*self.IDF(t) for t,tf in TFs.items()}
    
    def getTfsForStem(self,stem):
        '''Return the (doc-TF) representation of a stem from inverted-index'''
        return self.inverted_index[stem]
    
    def getTfsIDFsForStem(self,stem):
        '''Return the (doc-TFIDF) representation of a stem from inverted-index'''
        invTFs = self.getTfsForStem(stem)
        stemIDF = self.IDF(stem)
        return {d : tf*stemIDF for d,tf in invTFs.items()}
    
    def IDF(self,terme):
        ''' Compute IDF for terme '''
        return math.log((1+len(self.index)) / (1+len(self.inverted_index[terme])))

    ############# STATICS FUNCTIONS  ####################
    
    def write_dict(name,dico):
        with open(f"{IndexerSimple.save_folder}{name}.txt","w") as f:
            json.dump(dico,f,indent=4)
    
    def indexFromCorpus(corpus):
        '''Return index from given corpus'''
        index = dict()
        for doc in corpus.values() : 
            text = doc.data.get('T','')+' '+doc.getText()
            index[doc.id] = IndexerSimple.stem.getTextRepresentation(text)
        return index
    
    def invertIndex(index):
        d=dict()        
        for dico in index.values():
            d.update({key:dict() for key in dico.keys()})
        for w in d:
            for doc in index:
                if(w in index[doc].keys()):
                    d[w][doc] = index[doc][w]
                    
        return d        
   
    
