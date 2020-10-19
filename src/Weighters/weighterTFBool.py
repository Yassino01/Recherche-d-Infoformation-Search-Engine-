from Weighters.Weighter import Weighter

class WeigtherTFBool(Weighter):

    def getWeightsForDoc(self,idDoc):
        ''' returns weights for doc idDoc (w_td)'''
        return self.index[idDoc]
    
    def getWeightsForStem(self,stem):
        ''' returns weights of stem for all docs '''
        return self.inverted_index.get(stem,dict())

    def getWeightsForQuery(self,query):
        ''' returns weights of stem for all docs (w_tq)'''
        tf = self.stemmer.getTextRepresentation(query)
        return dict.fromkeys(tf.keys(),1)