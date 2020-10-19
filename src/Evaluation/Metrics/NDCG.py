from Evaluation.Metrics.EvalMetric import EvalMetric
from Evaluation.Metrics.F_Measure import F_Measure
import numpy as np 

class NDCG(EvalMetric):
    def __init__(self):
        self.F_measure = F_Measure()

    # TODO : generaliser pour avoir d'autre fonction de Relevance
    def relevance(doc, relevant_docs):
        ''' Heuristique pour calculer la pertinence d'un document sachant la liste de document pertinent'''
        return doc in relevant_docs

    def DCG(self,preds,relevant_docs):
        return sum( NDCG.relevance(doc,relevant_docs)/np.log2(i+2) 
                        for i,doc in enumerate(preds))
    
    def eval(self,preds,truth,**kwargs):
        return self.DCG(preds,truth)/self.DCG(truth,truth)        
    

