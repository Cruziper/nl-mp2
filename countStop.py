## Read and transform list of stopwords
special_characters = "\"!@#$%^&*()-+?_=,<>/\\..." ## Added ...
stopWordsF = open("stopwords.txt", "r")
string = stopWordsF.read()
fixedSTR = string.replace("\"", "\'")
elem = fixedSTR[2:-3]
swArray = elem.split("\', \'")

reviewsCorpus = []
with open("dev.txt", "r") as devF:
    for line in devF:
        reviewsCorpus.append(line.split("\t")[1])
        
reviewsSplit = []
reviewWordsLC = []
for review in reviewsCorpus:
    reviewsSplit.append(review.split(" "))
for review in reviewsSplit:
    for word in review:
        if word not in special_characters:
            reviewWordsLC.append(word.lower())

print(reviewWordsLC)

countWords = []
for stopWord in swArray:
    countWords.append(reviewWordsLC.count(stopWord))

print(countWords)

important_words = []
for index in range(len(countWords)):
    if countWords[index] == 1:
        important_words.append(swArray[index])
print("\n\n")
print(swArray)
print("\n")
print(important_words)