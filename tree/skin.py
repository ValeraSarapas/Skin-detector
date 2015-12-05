#Skin Segmentation Data Set from https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation
#face.png image from http://graphics.cs.msu.ru/ru/node/899

import numpy as np
import cv2

from sklearn import tree
from sklearn.cross_validation import train_test_split

#reads skin colors data
def ReadData():
    #Data in format [B G R Label] from
    data = np.genfromtxt('../data/Skin_NonSkin.txt', dtype=np.int32)

    labels= data[:,3]
    data= data[:,0:3]

    return data, labels

def BGR2HSV(bgr):
    bgr= np.reshape(bgr,(bgr.shape[0],1,3))
    hsv= cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2HSV)
    hsv= np.reshape(hsv,(hsv.shape[0],3))

    return hsv

#tworzy clasifier
def TrainTree(data, labels, flUseHSVColorspace):
    if(flUseHSVColorspace):
        data= BGR2HSV(data)

    #Split arrays or matrices into random train and test subsets
    trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.20, random_state=42)

    print trainData.shape
    print trainLabels.shape
    print testData.shape
    print testLabels.shape
    
    ##A decision tree classifier; The function to measure the quality of a split. Supported criteria are “gini” for the Gini
    #impurity and “entropy” for the information gain.
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    #Build a decision tree from the training set (X, y)
    clf = clf.fit(trainData, trainLabels)
    #Return the feature importances; The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance.
    print clf.feature_importances_
    #Returns the mean accuracy on the given test data and labels.
    print clf.score(testData, testLabels)

    return clf

def ApplyToImage(path, flUseHSVColorspace):
    data, labels= ReadData()#czytamy dane nt skory
    clf= TrainTree(data, labels, flUseHSVColorspace) #tworzymy clasifier

    img= cv2.imread(path)
    print img.shape
    data= np.reshape(img,(img.shape[0]*img.shape[1],3))
    print data.shape

    if(flUseHSVColorspace):#jesli wybrano HSV, konwertujemy do HSV
        data= BGR2HSV(data)

    #Predict class or regression value for X
    predictedLabels= clf.predict(data)

    imgLabels= np.reshape(predictedLabels,(img.shape[0],img.shape[1],1))#przeksztalcamy obraz tak by otrzymac zaznaczona skore

    if (flUseHSVColorspace):#zapisujemy wyniki
        cv2.imwrite('../results/result_HSV.png',((-(imgLabels-1)+1)*255))# from [1 2] to [0 255]
    else:
        cv2.imwrite('../results/result_RGB.png',((-(imgLabels-1)+1)*255))


#---------------------------------------------
fileName = '../example2.png'

print 'dla RGB:'
ApplyToImage(fileName, True)

print '\ndla HSV:'
ApplyToImage(fileName, False)