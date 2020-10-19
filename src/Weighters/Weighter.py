from abc import ABC,abstractmethod
from utils.TextRepresenter import PorterStemmer
import math

class Weighter(ABC) :
    
    """
    Les weighter transmettent : 
        --> les documents associé à un mot 
        --> les mots associé à un document 
        Le weighter ajoute une pondération pour indiquer le score de pertinence
        du terme dans un documment.
        Chacunes des classes enfants de weighter utilise un système de score 
        spécialisé.
    """
    
    def __init__(self,index,inverted_index):
        self.index = index
        self.inverted_index = inverted_index
        self.stemmer = PorterStemmer()

    @abstractmethod
    def getWeightsForDoc(self,idDoc):
        ''' returns weights for doc idDoc'''
        raise NotImplementedError
    
    @abstractmethod
    def getWeightsForStem(self,stem):
        ''' returns weights of stem for all docs'''
        raise NotImplementedError

    @abstractmethod
    def getWeightsForQuery(self,query):
        ''' returns weights of stem for all docs'''
        raise NotImplementedError
    
    
    def IDF(self,terme):
        ''' Compute IDF for terme '''
        return math.log((1+len(self.index)) / (1+len(self.inverted_index.get(terme,[]))))
