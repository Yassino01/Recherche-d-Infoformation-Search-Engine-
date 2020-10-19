
class Query:
    text_limit = 20
    rel_limit = 5
    
    def __init__(self,id,text=''):
        self.id = id
        self.text = text
        self.relevant_docs = []
        
    def addText(self,text):
        self.text+=text
        
    def addDoc(self,doc):
        self.relevant_docs.append(doc)
        
    def __repr__(self):
        return f"q({self.id}) = {self.text[:self.text_limit]}"+\
               f"{'...' if self.text_limit and len(self.text)>self.text_limit else ''}"+\
               f"\tr_docs = {self.relevant_docs[:self.rel_limit]}"+\
                f"{'...' if self.rel_limit and len(self.relevant_docs)>self.rel_limit else ''}\n"
            
    
        
        