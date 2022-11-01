import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import seaborn
import matplotlib.pyplot as plt

def plot_confusion_matrix(data, labels, output_filename):
    
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))
  
    seaborn.set(font_scale=1.5)
    ax = seaborn.heatmap(data, annot=True, cbar_kws={'label': 'Scale'}, fmt='g')
 
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
 
    ax.set(ylabel="True", xlabel="Predicted")
 
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()

def getConfusionMatrix(y_test, y_pred):
    # define data
    confusionMatrix = [[0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0]]
    for index in range(len(y_test)):
        # if they are equal
        if y_test[index] == y_pred[index]:
            if y_test[index] == "=Poor=":
                confusionMatrix[0][0] += 1
            elif y_test[index] == "=Unsatisfactory=":
                confusionMatrix[1][1] += 1
            elif y_test[index] == "=Good=":
                confusionMatrix[2][2] += 1
            elif y_test[index] == "=VeryGood=":
                confusionMatrix[3][3] += 1
            elif y_test[index] == "=Excellent=":
                confusionMatrix[4][4] += 1
        # if they are different
        else:
            # TRUTH : =Poor=
            if y_test[index] == "=Poor=" and y_pred[index] =="=Unsatisfactory=":
                confusionMatrix[0][1] += 1
            elif y_test[index] == "=Poor=" and y_pred[index] =="=Good=":
                confusionMatrix[0][2] += 1
            elif y_test[index] == "=Poor=" and y_pred[index] =="=VeryGood=":
                confusionMatrix[0][3] += 1
            elif y_test[index] == "=Poor=" and y_pred[index] =="=Excellent=":
                confusionMatrix[0][4] += 1
            # TRUTH : =Unsatisfactory=
            elif y_test[index] == "=Unsatisfactory=" and y_pred[index] =="=Poor=":
                confusionMatrix[1][0] += 1
            elif y_test[index] == "=Unsatisfactory=" and y_pred[index] =="=Good=":
                confusionMatrix[1][2] += 1
            elif y_test[index] == "=Unsatisfactory=" and y_pred[index] =="=VeryGood=":
                confusionMatrix[1][3] += 1
            elif y_test[index] == "=Unsatisfactory=" and y_pred[index] =="=Excellent=":
                confusionMatrix[1][4] += 1

            # TRUTH : =Good=
            elif y_test[index] == "=Good=" and y_pred[index] =="=Poor=":
                confusionMatrix[2][0] += 1
            elif y_test[index] == "=Good=" and y_pred[index] =="=Unsatisfactory=":
                confusionMatrix[2][1] += 1
            elif y_test[index] == "=Good=" and y_pred[index] =="=VeryGood=":
                confusionMatrix[2][3] += 1
            elif y_test[index] == "=Good=" and y_pred[index] =="=Excellent=":
                confusionMatrix[2][4] += 1

            # TRUTH : =VeryGood=
            elif y_test[index] == "=VeryGood=" and y_pred[index] =="=Poor=":
                confusionMatrix[3][0] += 1
            elif y_test[index] == "=VeryGood=" and y_pred[index] =="=Unsatisfactory=":
                confusionMatrix[3][1] += 1
            elif y_test[index] == "=VeryGood=" and y_pred[index] =="=Good=":
                confusionMatrix[3][2] += 1
            elif y_test[index] == "=VeryGood=" and y_pred[index] =="=Excellent=":
                confusionMatrix[3][4] += 1

            # TRUTH : =Excellent=
            elif y_test[index] == "=Excellent=" and y_pred[index] =="=Poor=":
                confusionMatrix[4][0] += 1
            elif y_test[index] == "=Excellent=" and y_pred[index] =="=Unsatisfactory=":
                confusionMatrix[4][1] += 1
            elif y_test[index] == "=Excellent=" and y_pred[index] =="=Good=":
                confusionMatrix[4][2] += 1
            elif y_test[index] == "=Excellent=" and y_pred[index] =="=VeryGood=":
                confusionMatrix[4][3] += 1
    return confusionMatrix

if __name__ == "__main__":

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

    reviewsLC = []
    for review in reviews:
        reviewsLC.append(review.lower())

    count_vect = CountVectorizer()
    count_matrix = count_vect.fit_transform(reviewsLC)
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

    clf = DecisionTreeClassifier(random_state=43)

    k_folds = KFold(n_splits = 5)
    sk_folds = StratifiedKFold(n_splits = 5)

    #scores = cross_val_score(clf, x, y, cv = k_folds)

    #print("Cross Validation Scores: ", scores)
    #print("Average CV Score: ", scores.mean())

    #cores = cross_val_score(clf, x, y, cv = sk_folds)

    #print("Cross Validation Scores: ", scores)
    #print("Average CV Score: ", scores.mean())

    #ss = ShuffleSplit(n_splits=5, test_size=0.11, random_state=42)
    #scores = cross_val_score(clf, x, y, cv=ss)

    #print("Cross Validation Scores: ", scores)
    #print("Average CV Score: ", scores.mean())

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.11, random_state=43)

    clf = svm.SVC(C=1).fit(x_train, y_train)
    #clf.score(x_test, y_test)

    scores = cross_val_score(clf, x, y, cv=5)
    y_pred = cross_val_predict(clf, x, y, cv=5)

    print("Cross Validation Scores: ", scores)
    print("Average CV Score: ", scores.mean())

    # define labels
    labels = ["1", "2", "3", "4", "5"]
    
    # create confusion matrix
    confusionMatrix = getConfusionMatrix(y_test, y_pred)
    plot_confusion_matrix(confusionMatrix, labels, "test.png")

    # Naive Bayes - scikit-learn library
    pass