import nltk
from nltk.corpus import stopwords # Not used
from nltk.stem import PorterStemmer # Not used
from nltk.stem import LancasterStemmer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB # Not used
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

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

    ## NOT in USE #############################
    stopwords = stopwords.words('english')
    porter = PorterStemmer()
    reviews2 = []
    ###########################################

    lancaster = LancasterStemmer()
    ranks = []
    reviews = []

    with open("train.txt", "r") as devF:
        for line in devF:
            rank_review = line.split("\t")
            ranks.append(rank_review[0])
            reviews.append(rank_review[1])


    reviewsLC = []
    for review in reviews:
        reviewsLC.append(review.lower())

    reviewsLCstem = []
    for review in reviewsLC:
        reviewsLCstem.append(lancaster.stem(review))

    count_vect = CountVectorizer()
    count_matrix = count_vect.fit_transform(reviewsLCstem)

    ## Data for classification
    with open("reviews.txt", "r") as rvF:
        for line in rvF:
            reviews2.append(line)
    count_matrix2 = count_vect.transform(reviews2)

    # Naive Bayes - scikit-learn library
    x = count_matrix
    y = ranks
    model = MultinomialNB()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.11, random_state=43, shuffle=True)
    y_pred = model.fit(x_train, y_train).predict(x_test)
    solF = open("results.txt", "w")
    for i in range(len(y_pred)):
        solF.write(y_pred[i])
        if i < len(y_pred)-1:
            solF.write("\n")
    accuracy = accuracy_score(y_test, y_pred)*100
    print(round(accuracy, 1))

    ## Classify new Data
    # pred_test = model.predict(count_matrix2)
    # solF = open("results.txt", "w")
    # for i in range(len(pred_test)):
    #     solF.write(pred_test[i])
    #     if i < len(pred_test)-1:
    #         solF.write("\n")

    # define labels
    labels = ["1", "2", "3", "4", "5"]
    
    # create confusion matrix
    confusionMatrix = getConfusionMatrix(y_test, y_pred)
    # plot_confusion_matrix(confusionMatrix, labels, "test.png")

    pass