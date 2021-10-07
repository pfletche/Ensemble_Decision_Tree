import time
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

def runDecisionTree(X_train,X_test,y_train,y_test):

    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    finishTime = time.time()
    print('Finished Training Decision Tree')

    y_pred = classifier.predict(X_test)

    return y_pred, finishTime

def runBaggingTree(X_train,X_test,y_train,y_test):

    bag_classifier = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.1)
    bag_classifier.fit(X_train, y_train)

    finishTime = time.time()

    y_predict = bag_classifier.predict(X_test)

    return y_predict, finishTime


