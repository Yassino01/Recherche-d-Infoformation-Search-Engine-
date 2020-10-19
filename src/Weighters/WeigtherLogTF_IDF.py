from Weighters.Weighter import Weighter
import math 

class WeigtherLogTF_IDF(Weighter):

    def getWeightsForDoc(self,idDoc):
        ''' returns weights for doc idDoc (w_td)'''
        stem_tf = self.index[idDoc].copy()
        stem_tf.map_values(lambda tf : 1 +math.log(tf ))
        return stem_tf
    
    def getWeightsForStem(self,stem):
        ''' returns weights of stem for all docs '''
        doc_weigth = dict()
        for doc,tf in self.inverted_index.get(stem,dict()):
            doc_weigth[doc] = 1+ math.log(tf)
        return doc_weigth

    def getWeightsForQuery(self,query):
        ''' returns weights of stem for all docs (w_tq)'''
        tf = self.stemmer.getTextRepresentation(query)
        query_weigth = dict()
        for stem in tf : 
            query_weigth[stem] = self.IDF(stem)
        return query_weigth
    