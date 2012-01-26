
class MatlabInputParser:
    
    def __init__(self1):
        # For required parameters
        self1.Rkeys = []
        self1.Rvalidators = []
        # For optional paramaters
        self1.Okeys = []
        self1.Odefaults = []
        self1.Ovalidators = []
        
        # Results
        self1.Results = dict([])
    
    def addRequired(self, name, validator):
        self.Rkeys.append(name)
        self.Rkeys.append(validator)
    
    def addParamValue(self, name, default, validator):
        self.Okeys.append(name)
        self.Odefaults
        self.Okeys.append(validator)
        
    def parse(self,indict):
        for key,validator in zip(self.Rkeys, self.Rvalidators):
            if key not in indict.keys():
                raise
            elif not validator(indict[key]):
                raise
            else:
                self.Results[key] = indict[key]
    
    
            
        