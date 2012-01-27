
class MatlabInputParser:
    
    def __init__(self):
        # For required parameters
        self.Rkeys = []
        self.Rvalidators = []
        # For optional paramaters
        self.Okeys = []
        self.Odefaults = []
        self.Ovalidators = []
        
        # Results
        self.Results = dict([])
    
    def addRequired(self, name, validator):
        self.Rkeys.append(name)
        self.Rvalidators.append(validator)
    
    def addParamValue(self, name, default, validator):
        self.Okeys.append(name)
        self.Odefaults.append(default)
        self.Ovalidators.append(validator)
        
    def parse(self,indict):
        for key,validator in zip(self.Rkeys, self.Rvalidators):
            if key not in indict.keys():
                raise
            elif not validator(indict[key]):
                raise
            else:
                self.Results[key] = indict[key]
                
        for key,default,validator in zip(self.Okeys, self.Odefaults, self.Ovalidators):
            if key not in indict.keys():
                self.Results[key] = default
            elif not validator(indict[key]):
                raise
            else:
                self.Results[key] = indict[key]                
    
    
            
        