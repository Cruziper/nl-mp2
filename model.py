import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

#print(stopwords.words('english'))

porter = PorterStemmer()
lancaster = LancasterStemmer()

ranks = []
reviews = []
reviews2 = []

with open("train.txt", "r") as devF:
    for line in devF:
        rank_review = line.split("\t")
        ranks.append(rank_review[0])
        reviews.append(rank_review[1])

with open("reviews.txt", "r") as rvF:
    for line in rvF:
        reviews2.append(line)

#with open("dev.txt", "r") as devF:
#    for line in devF:
#       rank_review = line.split("\t")
#        ranks.append(rank_review[0])
#        reviews.append(rank_review[1])

#print(ranks)
#print(reviews)

count_vect = CountVectorizer()
count_matrix = count_vect.fit_transform(reviews)
count_matrix2 = count_vect.transform(reviews2)

## Tokenized and lower case review words
#reviewsToken = [] # stores all reviews tokenized
#for review in reviews:
#    reviewsToken.append(review.split(" "))

#reviewsTokenLC = [] # stores all reviews tokenized and lower cased
#for review in reviewsToken:
#    tempReview = []
#    for token in review:
#        tempReview.append(token.lower())
#    reviewsTokenLC.append(tempReview)

#print("\nREVIEWS TOKENIZED --------------------------------------------------------------------------------")
#print(reviewsToken)

#print("\nTOKENS LOWER CASE --------------------------------------------------------------------------------")
#print(reviewsTokenLC)

## Word stemming
# Porter Stemming - least aggressive
#reviewsTokenLCstemP = [] # stores all reviews tokenized, lower cased and Porter stemmed
#for review in reviewsTokenLC:
#    tempReview = []
#    for token in review:
#        tempReview.append(porter.stem(token))
#    reviewsTokenLCstemP.append(tempReview)

#print("\n\nPORTER STEMMED --------------------------------------------------------------------------------")
#print(reviewsTokenLCstemP)


# Lancaster Stemming - least aggressive
#reviewsTokenLCstemL = [] # stores all reviews tokenized, lower cased and Lancaster stemmed
#for review in reviewsTokenLC:
#    tempReview = []
#    for token in review:
#        tempReview.append(lancaster.stem(token))
#    reviewsTokenLCstemL.append(tempReview)

#print("\n\nLANCASTER STEMMED --------------------------------------------------------------------------------")
#print(reviewsTokenLCstemL)

# Cross Validation - scikit-learn library

x = count_matrix
y = ranks

clf = DecisionTreeClassifier(random_state=42)

k_folds = KFold(n_splits = 5)

scores = cross_val_score(clf, x, y, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
clf.score(x_test, y_test)

scores = cross_val_score(clf, x, y, cv=5)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

# Naive Bayes - scikit-learn library