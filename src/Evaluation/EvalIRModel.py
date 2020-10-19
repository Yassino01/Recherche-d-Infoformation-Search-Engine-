import numpy as np
import pandas as pd
from scipy import stats

class EvalIRModel():
    '''
    Permet l'évaluation de différents modèles de recherche sur un ensemble de requête selon différentes mesures d'évaluation.
    Les résulats devront être résumé pour l'ensemble des requêtes considérée en présentant la moyenne et l'écart type pour chaques modèles.
    '''

    def compute_scores(self,model,metrics,queries,**metric_params):
        '''
        Test a model for each metrics each queries provided. 
        Parameters
        ----------
        model : IRModel object
        metrics : array of Metric objects
        queries : array of query objects
        metric_params : each metric can have special parameters (ex : rank =4 )

        Return a dict of arrays containing the score arrays for each metric 
        {'metric' : numpy_array , ... } 
        '''
        # test models
        scores = {str(metric) : np.zeros(len(queries)) for metric in metrics}
        for i,query in enumerate(queries) : 
            ranking = model.getRanking(query.text)
            docs,_ = zip(*ranking)
            docs = list(docs)
            for metric in metrics : 
                scores[str(metric)][i] = metric.evalQuery(docs,query,**metric_params)
        self.scores = scores

        # compute mean and variance :
        results = dict()
        for metric in metrics :
            results[str(metric)]={'mean': scores[str(metric)].mean()}
            results[str(metric)]['var'] = scores[str(metric)].var()
        self.df = pd.DataFrame(results)
        self.df.columns.name = str(model)

        return self.df


    def splitTrainTest(data,pct=0.5):
        assert 0 <= pct <=1   
        limit = int(round(len(data)*pct))
        assert 0 < limit < len(data)

        ids = np.arange(len(data))
        np.random.shuffle(ids)
        return data[ids[limit:]],data[ids[:limit]]


    def train_test_score(self,model,q_train,q_test,metric,gridSearch,**metric_params):
        ''' find best hyper-parameters for model compute train et test score on set of queries
        Parameters
        ----------
        model : IRModel object
        q_train : Query list
        q_test : Query list
        metric : Metric object
        '''
        params = gridSearch.getParams()
        scores = dict()
        
            # compute best param for train set 
        for i,param in enumerate(params):
            model.set_params(**param)
            df = self.compute_scores(model,[metric],q_train,**metric_params)
            scores[i] = df[str(metric)]['mean']
        best_param_id,train_score = max(scores.items(),key = lambda item : item[1])

            # compute score for test set
        best_param = params[best_param_id]
        model.set_params(**best_param)
        test_score = self.compute_scores(model,[metric],q_test,**metric_params)[str(metric)]['mean']
        
        return best_param,train_score,test_score

    def cross_validation_score(self,model,queries,metric,gridSearch,nb_folds=10,**metric_params):
        '''
            Cross validation score on K-folds 
            On each trains-folds, hyper parameters are optimized. Store score from test-fold.
            Parameters
            ----------
            model : IRModel object
            queries : Query list
            metric : Metric object
            gridSearch : ParamGrid object
            nb_folds : integer
            metric_params : additionnal params for metrics
            Returns array of scores
        '''
        ids = np.arange(len(queries))
        np.random.shuffle(ids)
        ids_batchs = np.array_split(ids,nb_folds)
        scores = np.zeros(nb_folds)
        for i in range(nb_folds):
            ids = np.hstack([ids_batchs[j] for j in range(nb_folds) if j!=i])
            q_train = queries[ids]
            q_test = queries[ids_batchs[i]]
            _,_,scores[i] = self.train_test_score(model,q_train,q_test,metric,gridSearch,**metric_params)
        return scores

    def t_test(self,model1,model2,queries,metric,rank=None, p_value=0.05):
        ''' Compute t-test between model1 and model2 on the list of queries using metric
        H0 : mean score M1 = mean score M2
        H1 : mean score M1 /= mean score M2
        return True if difference is significative (H1), false if not (H0)
        '''
        m1_pdf,m2_pdf = np.zeros(len(queries)),np.zeros(len(queries))
        for i,query in enumerate(queries):
            m1_pdf[i] =  metric.evalQuery(model1.getRankingDocs(query.text),query,rank=rank)
            m2_pdf[i] =  metric.evalQuery(model2.getRankingDocs(query.text),query,rank=rank)
        t,p = stats.ttest_ind(m1_pdf,m2_pdf)

        print(f"test passed for {p} value")

        return p > p_value


    




