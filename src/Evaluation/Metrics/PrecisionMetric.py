from Evaluation.Metrics.EvalMetric import EvalMetric

class PrecisionMetric(EvalMetric):

    def eval(self,pred,truth,**kwargs):
        return sum( el in truth for el in pred)/len(pred)
        
