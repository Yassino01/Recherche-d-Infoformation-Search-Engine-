from Evaluation.Query import Query
import numpy as np
import re

class QueryParser:
    
    
    def Parse(self,qry_fname,rel_fname,limit=None):
        '''Parse Query file Returns collection of Query '''
        
        self.loadQueryText(qry_fname,limit)
        self.loadQueryRelDocs(rel_fname,limit)
        
        # clean queries (removed queries without releveant_docs)
        self.queries = filter(lambda q : len(q.relevant_docs)>0,self.queries.values())
        self.queries = np.array(list(self.queries),dtype=object)
        return self.queries
    
    def loadQueryText(self,qry_fname,limit=None):
        ''' Create and load queries from files ''' 
        queries = dict()
        with open(qry_fname) as qry_f : 
           lines = qry_f.readlines(limit)
           curQuery = None
           tag = None
       
           for line in lines :
               match = re.match("\.([A-Z]) ?(\d+)?",line)
               if (match): # tag encountered
                   tag = match.groups()[0]
                   if (tag == "I"): # new query is created
                       identifier = int(match.groups()[1])
                       curQuery = Query(identifier)
                       queries[identifier] = curQuery
                       tag = None
               elif(tag == 'W'): #fill inside tag
                   curQuery.addText(line)
        
        self.queries = queries
           
                    
    def loadQueryRelDocs(self,rel_fname,limit=None):
        ''' load relevant document for all queries'''
        with open(rel_fname) as rel_f : 
            lines = rel_f.readlines(limit)        
            for line in lines :
                match = re.match("^ *(\d+) *(\d+).*",line)
                if (match): # tag encountered
                    identifier = int(match.groups()[0])
                    doc = int(match.groups()[1])
                    self.queries[identifier].addDoc(doc)    