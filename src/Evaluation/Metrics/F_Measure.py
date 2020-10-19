from Evaluation.Metrics.EvalMetric import EvalMetric
from Evaluation.Metrics.PrecisionMetric import PrecisionMetric
from Evaluation.Metrics.RecallMetric import RecallMetric


class F_Measure(EvalMetric):

    def __init__(self):
        self.precision = PrecisionMetric()
        self.recall = RecallMetric()

    def eval(self,pred,truth,beta = 1,**kwargs):
        p = self.precision.eval(pred,truth)
        r = self.recall.eval(pred,truth)
        if (p == 0 and r == 0):
            return 0
        return (1+beta**2) *( p*r) / ((beta**2)*p + r)
        

        
