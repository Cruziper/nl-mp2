import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

stopWords = stopwords.words('english')

porter = PorterStemmer()
lancaster = LancasterStemmer()

ranks = []
reviews = []
reviews2 = []

def inputToMatrix (reviews):
    ## Tokenized and lower case review words
    reviewsToken = [] # stores all reviews tokenized
    for review in reviews:
        reviewLower = review.lower()
        reviewToken = word_tokenize(reviewLower)
        reviewsToken.append(reviewToken)

    reviewsTokenLC = [] # stores all reviews tokenized and lower cased
    for review in reviewsToken:
        tempReview = []
        for token in review:
            if token.lower() not in stopWords:
                tempReview.append(token.lower())
        reviewsTokenLC.append(tempReview)

    ## Word stemming
    # Porter Stemming - least aggressive
    reviewsTokenLCstemP = [] # stores all reviews tokenized, lower cased and Porter stemmed
    for review in reviewsTokenLC:
        tempReview = []
        for token in review:
            tempReview.append(porter.stem(token))
        reviewsTokenLCstemP.append(tempReview)

    tokenTags = []
    for review in reviewsToken:
        tags = nltk.pos_tag(review, tagset="universal")
        tokenTags.append(tags)

    ## matrix #noun / #adj / #det / #verb / #pron / #adv / #conj / #adp / #prt / #num / #ponct / #other
    matrixPOScount = []
    for review in tokenTags:
        nmbNOUN = 0
        nmbADJ = 0
        nmbDET = 0
        nmbVERB = 0
        nmbPRON = 0
        nmbADV = 0
        nmbCONJ = 0
        nmbADP = 0
        nmbPRT = 0
        nmbNUM = 0
        nmbPONCT = 0
        nmbOTHER = 0
        nmbSW = 0
        for token in review:
            if 'NOUN' in token:
                nmbNOUN += 1
            elif 'ADJ' in token:
                nmbADJ += 1
            elif 'DET' in token:
                nmbDET += 1
            elif 'VERB' in token:
                nmbVERB += 1
            elif 'PRON' in token:
                nmbPRON += 1
            elif 'ADV' in token:
                nmbADV += 1
            elif 'CONJ' in token:
                nmbCONJ += 1
            elif 'ADP' in token:
                nmbADP += 1
            elif 'PRT' in token:
                nmbPRT += 1
            elif 'NUM' in token:
                nmbNUM += 1
            elif '.' in token:
                nmbPONCT += 1
            elif 'X' in token:
                nmbOTHER += 1
            else:
                print(token)
        matrixPOScount.append([nmbNOUN, nmbADJ, nmbVERB, nmbPRON, nmbADV, nmbCONJ, nmbADP, nmbPRT, nmbNUM, nmbPONCT, nmbOTHER])
    return matrixPOScount

with open("train.txt", "r") as devF:
    for line in devF:
        rank_review = line.split("\t")
        ranks.append(rank_review[0])
        reviews.append(rank_review[1])

with open("reviews.txt", "r") as rvF:
    for line in rvF:
        reviews2.append(line)

# print(ranks)
# print(reviews)

matrix1 = inputToMatrix(reviews)
matrix2 = inputToMatrix(reviews2)
print(matrix2)

# print("\n\nPORTER STEMMED --------------------------------------------------------------------------------")
# print(reviewsTokenLCstemP)

# count_vect = CountVectorizer()
# count_matrix = count_vect.fit_transform(reviews)
# count_array = count_matrix.toarray()
# df = pd.DataFrame(data=count_array,columns = count_vect.get_feature_names())
# print(df)

# # Lancaster Stemming - least aggressive
# reviewsTokenLCstemL = [] # stores all reviews tokenized, lower cased and Lancaster stemmed
# for review in reviewsTokenLC:
#     tempReview = []
#     for token in review:
#         tempReview.append(lancaster.stem(token))
#     reviewsTokenLCstemL.append(tempReview)

# print("\n\nLANCASTER STEMMED --------------------------------------------------------------------------------")
# print(reviewsTokenLCstemL)

# Naive Bayes - scikit-learn library
x = matrix1
y = ranks
model = MultinomialNB()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=15, shuffle=True)
y_pred = model.fit(x_train, y_train).predict(x_test)
# solF = open("results.txt", "w")
# for i in range(len(y_pred)):
#     solF.write(y_pred[i])
#     if i < len(y_pred)-1:
#         solF.write("\n")
accuracy = accuracy_score(y_test, y_pred)*100

y_pred = model.predict(matrix2)
solF = open("results.txt", "w")
for i in range(len(y_pred)):
    solF.write(y_pred[i])
    if i < len(y_pred)-1:
        solF.write("\n")

print(accuracy)
