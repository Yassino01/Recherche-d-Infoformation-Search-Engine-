from IRModels.IRModel import IRModel 
import math
import numpy as np
class Vectoriel(IRModel):
    
    def __init__(self,weighter,inner_product):
        super().__init__(weighter)
        self.inner_product = inner_product
    
    def set_params(self,inner_product):
        self.inner_product = inner_product

    def getScores(self,query):
        scores = dict()
        query_weigths = self.weighter.getWeightsForQuery(query)
        
        for stem in query_weigths :
            doc_tf = self.weighter.getWeightsForStem(stem)
            for doc in doc_tf:
                if (doc not in scores.keys()):
                    scores[doc] = 0
                inner_product = query_weigths[stem] *doc_tf[doc]
                scores[doc]+= inner_product
        
        if (not inner_product) :# si c'est la mesure de cosinus on normalise
            Y_norm = sum(w**2 for w in query_weigths.values())
            for doc in scores : 
                X_norm = sum(w**2 for w in self.weighter.getWeightsForDoc(doc).values)
                denominator = math.sqrt(X_norm)+math.sqrt(Y_norm)
                scores[doc]  =  scores[doc]/ denominator
        return scores