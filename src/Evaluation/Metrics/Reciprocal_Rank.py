from Evaluation.Metrics.EvalMetric import EvalMetric

class Reciprocal_Rank(EvalMetric):

    # Attention : 
    # ici on suppose que la liste truth est ordonnée
    def eval(self,prediction,truth,**kwargs):
        for doc in prediction : 
            if doc in truth:
                return 1/(truth.index(doc)+1) 
        return 0
        
        
