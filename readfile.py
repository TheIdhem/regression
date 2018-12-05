
class Readfile:
    def __init__(self, fileName):
        self.fileName = fileName

    def readDataCol(self, column):
        df = open(self.fileName, "r")
        lines = df.readlines()
        result = []
        for x in lines:
            result.append(float(x.split()[column-1]))
        df.close()
        return result


    def readDataLines(self):
        df = open(self.fileName, "r")
        lines = df.readlines()
        result = []
        for x in lines:
            line = []
            line = x.split()
            line = map(float, line)
            line.pop()
            result.append(line)
        return result


    def readDataLines(self):
        df = open(self.fileName, "r")
        lines = df.readlines()
        result = []
        for x in lines:
            line = []
            line = x.split()
            line = map(float, line)
            line.pop()
            result.append(line)
        return result
