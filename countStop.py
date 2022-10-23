## Read and transform list of stopwords
file = open("stopwords.txt", "r")
string = file.read()
fixedSTR = string.replace("\"", "\'")
elem = fixedSTR[2:-3]
elems = elem.split("\', \'")