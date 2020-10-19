from Evaluation.Metrics.EvalMetric import EvalMetric

class RecallMetric(EvalMetric):

    def eval(self,pred,truth,**kwargs):
        return sum( el in truth for el in pred)/len(truth)
 
        
