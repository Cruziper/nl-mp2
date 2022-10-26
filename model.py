import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

print(stopwords.words('english'))

porter = PorterStemmer()
lancaster = LancasterStemmer()

ranks = []
reviews = []

with open("dev.txt", "r") as devF:
    for line in devF:
        rank_review = line.split("\t")
        ranks.append(rank_review[0])
        reviews.append(rank_review[1])

print(ranks)
print(reviews)

## Tokenized and lower case review words
reviewsToken = [] # stores all reviews tokenized
for review in reviews:
    reviewsToken.append(review.split(" "))

reviewsTokenLC = [] # stores all reviews tokenized and lower cased
for review in reviewsToken:
    tempReview = []
    for token in review:
        tempReview.append(token.lower())
    reviewsTokenLC.append(tempReview)

print("\nREVIEWS TOKENIZED --------------------------------------------------------------------------------")
print(reviewsToken)

print("\nTOKENS LOWER CASE --------------------------------------------------------------------------------")
print(reviewsTokenLC)

## Word stemming
# Porter Stemming - least aggressive
reviewsTokenLCstemP = [] # stores all reviews tokenized, lower cased and Porter stemmed
for review in reviewsTokenLC:
    tempReview = []
    for token in review:
        tempReview.append(porter.stem(token))
    reviewsTokenLCstemP.append(tempReview)

print("\n\nPORTER STEMMED --------------------------------------------------------------------------------")
print(reviewsTokenLCstemP)


# Lancaster Stemming - least aggressive
reviewsTokenLCstemL = [] # stores all reviews tokenized, lower cased and Lancaster stemmed
for review in reviewsTokenLC:
    tempReview = []
    for token in review:
        tempReview.append(lancaster.stem(token))
    reviewsTokenLCstemL.append(tempReview)

print("\n\nLANCASTER STEMMED --------------------------------------------------------------------------------")
print(reviewsTokenLCstemL)

# Cross Validation - scikit-learn library

# Naive Bayes - scikit-learn library