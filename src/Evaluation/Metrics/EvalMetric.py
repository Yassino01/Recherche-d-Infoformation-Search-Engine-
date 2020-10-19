from abc import ABC,abstractmethod

class EvalMetric(ABC):
    
    @abstractmethod
    def eval(self,prediction,truth,**kwargs):
        raise NotImplementedError()

    def evalQuery(self,prediction,query,rank=None,**kwargs):
        '''return score of the ranking of a query.
         @prediction : array of id of docs
         @query : object of type query
         @rank : int  score only perform on the @rank first results
         '''
        return self.eval(prediction[:rank], query.relevant_docs[:rank],**kwargs) 

    def __str__(self):
        return self.__class__.__name__
  
        
            
    
        
        