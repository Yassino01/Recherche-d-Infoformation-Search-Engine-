from Evaluation.Metrics.EvalMetric import EvalMetric

class AveragePrecision(EvalMetric):
    
    def eval(self,preds,truth,**kwargs):
        ap = 0
        tp = 0 # true positive 
        for i,p in enumerate(preds) : 
            if (p in truth):
                tp = tp+1
                ap+= tp/(i+1)
        return ap/len(truth)
        
            
    
        
        