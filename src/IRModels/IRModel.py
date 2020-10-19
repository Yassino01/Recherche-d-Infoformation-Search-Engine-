# Information Research Model 
from abc import ABC,abstractmethod

class IRModel(ABC) : 
    """
        Les Information Research Model : 
        Permettent de récupérer un ensemble de document 
        pertinents à partir d'une requete.
    """
    def __init__(self,weighter):
        self.weighter = weighter
    
    @abstractmethod
    def set_params(self,**kwargs):
        ''' set custom params for model '''
        raise NotImplementedError()

    @abstractmethod
    def getScores(self,query):
        '''Returns dict with best docs associated with scores'''
        raise NotImplementedError()
        
    def getRanking(self,query):
        ''' return array of sorted couples {doc : score} from best to worst'''
        score = self.getScores(query)
        return sorted(score.items(),reverse=True,key= lambda x: x[1])
    
    def getRankingDocs(self,query):
        '''returns array of docs, from best to worst'''
        scores = self.getScores(query)
        return sorted(scores.keys(),reverse=True,key= lambda k: scores[k])
    
    def __str__(self):
        return self.__class__.__name__