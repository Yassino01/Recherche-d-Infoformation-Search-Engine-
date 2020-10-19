from Indexation.Parser import Parser
from Indexation.IndexerSimple import IndexerSimple
from Weighters.WeighterTFTF import WeighterTFTF
from IRModels.Vectoriel import Vectoriel
from IRModels.ModeleLangue import ModeleLangue
from IRModels.Okapi import Okapi
from Evaluation.QueryParser import QueryParser
from Evaluation.EvalIRModel import EvalIRModel 
from Evaluation.Metrics import * 
from Evaluation.ParamGrid import ParamGrid
import numpy as np

    
def testTME1(indexer):
    print("\n\tSTART TEST for TME1")
    doc = 1
    print (f"raw text for doc {doc}: ")
    print(indexer.getStrDoc(doc)[:30],'...')
    print(f"TF from doc {doc}: ")
    print (list(indexer.getTfsForDoc(doc).items())[:5],'...')
    print(f"TF-IDF from doc {doc}")
    print(list(indexer.getTfsIDFsForDoc(doc).items())[:5],'...')
    
    stem = "preliminari"
    print(f"TF-Doc from stem {stem}")
    print(list(indexer.getTfsForStem(stem).items())[:5],'...')
    print(f"TFIDF-Doc from stem {stem}")
    print(list(indexer.getTfsIDFsForStem(stem).items())[:5],'...')
    
    print("END TEST for TME1")

def testTME2(indexer):
    print("\n\tSTART TEST for TME2")
    doc = 1
    print("\nTest Weighter : ")
    w_tftf = WeighterTFTF(indexer.index,indexer.inverted_index)
    print(f"test tftf weight for {doc}")
    print(w_tftf.getWeightsForDoc(doc))
    
    print("\nTest Inner Product Modele Vectoriel")
    query = "algebra report"
    mVec = Vectoriel(w_tftf,False)
    bestDoc,score = mVec.getRanking(query)[0]
    print(f"for query : {query}\nbestResult : doc_id={bestDoc},score={score}\n best doc text : \n{indexer.getStrDoc(bestDoc)[:40]}...")
    
    print("\nTest modele de langue")
    mLangue = ModeleLangue(w_tftf)
    bestDoc,score = mLangue.getRanking(query)[0]
    print(f"for query : {query}\nbestResult : doc_id={bestDoc},score={score}\n best doc text : \n{indexer.getStrDoc(bestDoc)[:40]}")
    
    print("\nTest Okapi BM-25")
    mOkapi = Okapi(w_tftf)
    bestDoc,score = mOkapi.getRanking(query)[0]
    print(f"for query : {query}\nbestResult : doc_id={bestDoc},score={score}\n best doc text : \n{indexer.getStrDoc(bestDoc)[:40]}")
    
    print("END TEST for TME2")

def testTME3(indexer,path_qry,path_rel):
    print("\t\tSTART TEST for TME3")
    print("\tPART 1 : LOAD QUERY DATABASE")
    Qparser = QueryParser()
    queries = Qparser.Parse(path_qry, path_rel)
    print([queries[i] for i in range(1,4)])

    ## create models : 
    w_tftf = WeighterTFTF(indexer.index,indexer.inverted_index)
    mVec = Vectoriel(w_tftf,False)
    okapiModel = Okapi(w_tftf)
    mLangue = ModeleLangue(w_tftf)


    print("\tPart 2 : Test Metrics :")
        # load query for model
    
    query = queries[0]
    ranking = mVec.getRanking(query.text)
    docs,_ = zip(*ranking) # equivalent of unzip
    docs = list(docs)
        # try metrics : 
    metrics = [PrecisionMetric(),RecallMetric(),F_Measure(),NDCG(),AveragePrecision(),Reciprocal_Rank()]
    print(f"query : {query.text}")
    for m in metrics : 
        print(f"{type(m).__name__} :", m.evalQuery(docs,query,rank=None)) 

    print("\n\tPart 3 : Evaluation :")
    print("No rank limit : ")
    evalIRModel = EvalIRModel()
    evalIRModel.compute_scores(mVec,metrics,queries,rank=None)
    print(evalIRModel.df)
    evalIRModel.compute_scores(okapiModel,metrics,queries,rank=None)
    print(evalIRModel.df)
    evalIRModel.compute_scores(mLangue,metrics,queries,rank=None)
    print(evalIRModel.df)


    print("\n\t Part 4 : Bonus")
    print("\n\t\t T_test")
    significative = evalIRModel.t_test(mVec,mLangue,queries,Reciprocal_Rank())
    print(f"the diffence beeween modelLangue and okapi is significative : {significative}")

    print("\n\t\t param_optimizer train/test with AveragePrecision")
    print("Test for mLangue")
    grid = ParamGrid({'l_short': [0,1] ,'l_long': [0,1]})
    q_train,q_test = EvalIRModel.splitTrainTest(queries[:30],0.5)
    best_param,train_score,test_score = evalIRModel.train_test_score(mLangue,q_train,q_test,AveragePrecision(),grid)
    print(f'best_param : {best_param},train_score: {round(train_score,4)},test_score: {round(test_score,4)}')
    print("\nTest for OkapiBM25")
    grid = ParamGrid({'b': [0,1],'k1': [0,3] })
    q_train,q_test = EvalIRModel.splitTrainTest(queries[:30],0.5)
    best_param,train_score,test_score = evalIRModel.train_test_score(okapiModel,q_train,q_test,AveragePrecision(),grid)
    print(f'best_param : {best_param},train_score: {round(train_score,4)},test_score: {round(test_score,4)}')

    print("\n\t\tCross validation score with AveragePrecision")
    print("for OkapiBM25")
    grid = ParamGrid({'b': [0,1],'k1': [0,3] })
    q_train,q_test = EvalIRModel.splitTrainTest(queries[:30],0.5)
    scores = evalIRModel.cross_validation_score(okapiModel,queries,AveragePrecision(),grid,nb_folds=5)
    print(f"mean score : {scores.mean()}")
    print(f"all score : {scores}")


def main():
    print("Start")

    cacm = {"name":"cacm", "path_txt" : "../data/cacm/cacm.txt",
            "path_qry" : "../data/cacm/cacm.qry","path_rel" : "../data/cacm/cacm.rel"}
    cisi = {"name":"cisi", "path_txt" : "../data/cisi/cisi.txt",
             "path_qry" : "../data/cisi/cisi.qry", "path_rel" : "../data/cisi/cisi.rel" }
    
    fileDict = cacm
    corpus = Parser.Parse(fileDict["path_txt"])
    print(f"{fileDict['name']} loaded")
    indexer = IndexerSimple()
    indexer.indexation(corpus,fileDict["name"])
    
    print(f"{fileDict['name']} indexed")
    
    testTME1(indexer)
    testTME2(indexer)
    testTME3(indexer,fileDict['path_qry'],fileDict['path_rel'])

if(__name__ == "__main__"):
    main()
    
    
    