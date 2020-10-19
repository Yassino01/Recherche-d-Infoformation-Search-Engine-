class Document:
    
    def __init__(self,identifier):
        self.id = identifier
        self.data = dict()
        
    def addToTag(self,tag,content): 
        if tag not in self.data:
            self.data[tag] = ""
        self.data[tag]+= content
        
    def getText(self):
        return self.data.get('W',"")
    
    def __str__(self):
        return f"Document {self.id}:\n{self.data}"
    
    def __repr__(self):
        return self.__str__()