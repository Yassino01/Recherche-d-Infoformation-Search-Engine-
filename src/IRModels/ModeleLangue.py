#modele de langue (lissage Jelinek-Mercer)
from IRModels.IRModel import IRModel 
import numpy as np

class ModeleLangue (IRModel):
    
    LongQuerySize = 15
    def __init__(self,weighter):
        super().__init__(weighter)
        
            # calcul du TF(ou autre mesure) total pour chaque doc.
        self.nbTDocs = { doc : sum(tfs.values()) for doc,tfs in weighter.index.items()}
            # calcul du TD(ou autre mesure) total pour le corpus.
        self.nbTCorpus = sum(self.nbTDocs.values())

        self.l_short = 0.8 
        self.l_long = 0.2
    
    def set_params(self,l_short,l_long):
        self.l_short = l_short
        self.l_long = l_long

    def getScores(self,query):
        scores = dict()
        q_weigths = self.weighter.getWeightsForQuery(query)
        
            # choix de param lambda 
        if(len(q_weigths)<ModeleLangue.LongQuerySize):
            l = self.l_short
        else:
            l = self.l_long
        
        for stem in q_weigths:
            for docId,doc_tf in self.weighter.getWeightsForStem(stem).items():
                if (docId not in scores):
                    scores[docId] = 1
                p_t_md = doc_tf/ self.nbTDocs[docId] # proba obtenir t sur doc d
                p_t_mc = doc_tf / self.nbTCorpus # proba obtenir t sur corpus
                scores[docId] *=(l * p_t_md + (1-l) * p_t_mc ) * q_weigths[stem]
            
        return scores
    