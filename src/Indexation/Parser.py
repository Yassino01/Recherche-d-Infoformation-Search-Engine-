from Indexation.Document import Document
import re

class Parser:
        
    def __init__(self):
        pass
    
    def Parse(filename,limit = -1):
        '''
        Parse file and return corpus (dictionnary of documents)
        cacm.txt format style is expected
        '''
        corpus = dict()
        
        with open(filename) as file:
            lines = file.readlines(limit)   
            curDoc = None
            tag = None
        
            for line in lines :
                match = re.match("\.([A-Z]) ?(\d+)?",line)
                if (match): # tag encountered
                    tag = match.groups()[0]
                    if (tag == "I"): # new doc is created
                        identifier = int(match.groups()[1])
                        curDoc = Document(identifier)
                        corpus[identifier] = curDoc
                        tag = None
                elif(tag): #fill inside tag
                    curDoc.addToTag(tag,line)
        return corpus
    
    