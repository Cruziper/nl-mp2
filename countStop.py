## Read and transform list of stopwords
file = open("stopwords.txt", "r")
string = file.read()
fixedSTR = string.replace("\"", "\'")
print(fixedSTR[0])
print(fixedSTR[1])
print(fixedSTR[2])
print(fixedSTR[3])
elem = fixedSTR[2:-3]
elems = elem.split("\', \'")
print(elems[0])
print(elems[1])
print(elems[2])
print(elems[3])