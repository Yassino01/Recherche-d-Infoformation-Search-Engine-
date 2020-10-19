from IRModels.IRModel import IRModel 
import numpy as np
# Okapi-BM25 model
class Okapi(IRModel):
    
    def __init__(self,weighter):
        ''' En principe il faudrait weighterTFTF en paramètre '''
        super().__init__(weighter)
        self.k1 = 1.2# cf cours
        self.b = 0.75

            # calcul du TF(ou autre mesure) total pour chaque doc.
        self.nbTDocs = { doc : sum(tfs.values()) for doc,tfs in weighter.index.items()}
            # à calculer :
        self.avgdl = np.mean(list(self.nbTDocs.values())) #average document length
    
    def set_params(self,k1,b):
        self.k1 = k1
        self.b = b

    def getScores(self,query):
        scores = dict()
        q_weigths = self.weighter.getWeightsForQuery(query)

        for stem in q_weigths:
            q_tf = q_weigths[stem]
            idf = self.weighter.IDF(stem)
            for docId,doc_tf in self.weighter.getWeightsForStem(stem).items():
                docL = self.nbTDocs[docId]
                if(docId not in scores):
                    scores[docId] = 0
                scores[docId]+= q_tf * idf * (doc_tf / (doc_tf +self.k1*(1-self.b+self.b*docL/self.avgdl)))
        
        return scores
    