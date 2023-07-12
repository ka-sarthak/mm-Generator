import datetime

class Logger():
    def __init__(self,filename):
        self.file = open(filename,"w")
        self.startTime = datetime.datetime.now()
        self.addLine(f"Starting the log")
        
    def addLine(self,message):
        self.file.write(f"{datetime.datetime.now()}: {message}\n")
        
    def addRow(self,row):
        self.file.write(f"{datetime.datetime.now()}: ")
        self._addRow(row)
    
    def _addRow(self,row):
        for el in row:
            self.file.write(f"\t{round(el,6)}")
        self.file.write("\n")
        
    def addTable(self,table,title):
        self.addLine(f"Logging the table titled - {title}")
        self._addTable(table)
        
    def _addTable(self,table):
        for row in table:
            self._addRow(row)
        
    def close(self):
        self.endTime = datetime.datetime.now()
        self.addLine(f"Ending the log - Logged for {round(self.endTime.timestamp()-self.startTime.timestamp(),6)} seconds")
        self.file.close()
        