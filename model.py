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

# print(stopwords.words('english'))

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

# print(ranks)
# print(reviews)

count_vect = CountVectorizer()
count_matrix = count_vect.fit_transform(reviews)
# count_array = count_matrix.toarray()
# df = pd.DataFrame(data=count_array,columns = count_vect.get_feature_names())
# print(df)

count_matrix2 = count_vect.transform(reviews2)




# ## Tokenized and lower case review words
# reviewsToken = [] # stores all reviews tokenized
# for review in reviews:
#     reviewLower = review.lower()
#     reviewToken = word_tokenize(reviewLower)
#     reviewsToken.append(reviewToken)

# tokenTags = []
# for review in reviewsToken:
#     tags = nltk.pos_tag(review, tagset="universal")
#     tokenTags.append(tags)
#     print(tokenTags)

# reviewsTokenLC = [] # stores all reviews tokenized and lower cased
# for review in reviewsToken:
#     tempReview = []
#     for token in review:
#         tempReview.append(token.lower())
#     reviewsTokenLC.append(tempReview)

# print("\nREVIEWS TOKENIZED --------------------------------------------------------------------------------")
# print(reviewsToken)

# print("\nTOKENS LOWER CASE --------------------------------------------------------------------------------")
# print(reviewsTokenLC)

# ## Word stemming
# # Porter Stemming - least aggressive
# reviewsTokenLCstemP = [] # stores all reviews tokenized, lower cased and Porter stemmed
# for review in reviewsTokenLC:
#     tempReview = []
#     for token in review:
#         tempReview.append(porter.stem(token))
#     reviewsTokenLCstemP.append(tempReview)

# print("\n\nPORTER STEMMED --------------------------------------------------------------------------------")
# print(reviewsTokenLCstemP)


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
x = count_matrix
y = ranks
model = MultinomialNB()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)
y_pred = model.fit(x_train, y_train).predict(x_test)
solF = open("results.txt", "w")
for i in range(len(y_pred)):
    solF.write(y_pred[i])
    if i < len(y_pred)-1:
        solF.write("\n")
accuracy = accuracy_score(y_test, y_pred)*100

pred_test = model.predict(count_matrix2)

print(accuracy)
