from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier,export_text
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import csv
import pathlib
import mpld3
from mpld3 import plugins


def load_data(filename):
    with open(filename,'r') as infile:  # Reads the file
        reader = csv.reader(infile) # Used to grab labels
        colNames = next(reader) # Iterates and store labels
        rows = np.array(list(reader), dtype=np.float64) # Store Data and Cast to float64
    return colNames, rows

def seperate_labels(colNames, rows):
    labelCol = colNames.index('Outcome')    # label index
    ys = rows[:,labelCol]   # outcomes
    xs = np.delete(rows,labelCol, axis=1)   # deleting outcome column from data
    del colNames[labelCol]  # Removes outcome from label names
    return colNames, xs, ys

def preprocess_data(colNames, xs):
    bloodGlucoseIndex = 2
    newfeature = (xs[:,bloodGlucoseIndex]<1.0).reshape((xs.shape[0],1))
    xs = np.concatenate((xs,newfeature),axis=1)
    colNames.append('NoBloodGlucoseData')
    return colNames,xs

def write_data(filename, ypred):
   with open(filename,'w') as outfile:  # Reads the file
        writer = csv.writer(outfile, lineterminator = '\n') # Used to grab labels
        writer.writerow(['Outcome'])
        for i in ypred:
            writer.writerow([i])


# def measurePerformance(clf,X_test,Y_test):
#     scores = cross_val_score(clf, X_test, Y_test,cv=5)
#     print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#     return scores


# def hyperparameter(x_train, x_test, y_train, y_test):
#     training_acc = []
#     testing_acc = []
#     x = range(1,20)
#     for i in x:
#         classifier = KNeighborsClassifier(n_neighbors=i).fit(x_train, y_train)
#         train_acc = classifier.score(x_train, y_train)
#         test_acc = classifier.score(x_test, y_test)
#         training_acc.append(train_acc)
#         testing_acc.append(test_acc)
#         print("Training_Accuracy:\t", train_acc,"\tTest_Accuracy\t", test_acc)
#     plt.plot(x, training_acc, label='training accuracy')
#     plt.plot(x, testing_acc, label='test accuracy')
#     plt.title("KNeighborsClassifier accuracy with n_neighbors")
#     plt.xlabel('n_neighbors')
#     plt.ylabel('accuracy')
#     plt.legend()
#     plt.show()
#     return



random_state = 42
path = pathlib.Path(__file__).parents[1]
# print("\n\n\n\HERERERRER" ,path)
colNames, data = load_data(path/'saved_models//diabetes.csv')
colNames, xs, ys = seperate_labels(colNames,data)
#colNames, xs = preprocess_data(colNames,xs)

xtrain, xtest, ytrain, ytest = train_test_split(xs,ys,test_size=0.25,random_state=random_state)

# clf = DecisionTreeClassifier(random_state=random_state).fit(xtrain,ytrain)
clf1 = svm.SVC(random_state=random_state).fit(xtrain,ytrain) 
# clf2 = LinearDiscriminantAnalysis().fit(xtrain,ytrain)
# clf3 = GaussianNB().fit(xtrain,ytrain)
# kneigh = KNeighborsClassifier(n_neighbors=16).fit(xtrain,ytrain)

#hyperparameter(xtrain, xtest, ytrain, ytest)
#print(export_text(clf,feature_names = colNames))

# ypred_dt = clf.predict(xtest)
# print("Decision Tree\nAccuracy:\t", accuracy_score(ytest,ypred_dt))
# print(confusion_matrix(ytest,ypred_dt))

ypred_svm = clf1.predict(xtest)
# print("\nSVM\nAccuracy:\t",accuracy_score(ytest,ypred_svm ))
# print(confusion_matrix(ytest,ypred_svm ))

# ypred_LDA = clf2.predict(xtest)
# print("\nLDA\nAccuracy:\t",accuracy_score(ytest,ypred_LDA))
# print(confusion_matrix(ytest,ypred_LDA))

# ypred_b = clf3.predict(xtest)
# print("\nBayes\nAccuracy:\t",accuracy_score(ytest,ypred_b))
# print(confusion_matrix(ytest,ypred_b))

# ypred_n = kneigh.predict(xtest)
# print("\nKneigh\nAccuracy:\t",accuracy_score(ytest,ypred_n))
# print(confusion_matrix(ytest,ypred_n))

def run_algo1(file_path, debug=False):
    
    colNames, data = load_data(file_path)
    #colNames, data = preprocess_data(colNames,data)
    ypred = clf1.predict(data)
    # Load file_path instead of unknowns.csv
    # Do the same processing you did to
    # get your unknowns.csv results from your model.
    # Get the figure object Matplotlib will be working with:
    fig = plt.figure()
    plt.scatter(data[ypred == 1,1],data[ypred == 1,7])
    plt.scatter(data[ypred != 1,1],data[ypred != 1,7])
    plt.title("Predicted and Missed Predicted")
    plt.legend(["Diabetic","Non Diabetic"])
    plt.xlabel('Glucose')
    plt.ylabel('Age')
    # if debug:
    #     plt.show()
    # Scatterplot your results, as per
    # any of your scatterplots from homework 1.
    # Make sure to use plt.xlabel(), plt.ylabel()
    # and plt.title() to give clear labeling!
    plugins.connect(fig, plugins.MousePosition(fontsize=12, fmt='.3g'))
    plugins.connect(fig, plugins.Zoom(button=True, enabled=None))

    return mpld3.fig_to_html(fig)


# if __name__=="__main__":
#     run_algo1("C:\\Users\\Edwin\\OneDrive\\Documents\\UCSB\\ECE 157A\\Final_Project_Phase1\\Final_Project_Phase1\\Final_Project\\media\\upload\\unknowns.csv", False)


#ypred = clf1.predict(xtest)
#plt.scatter(xtest[ypred == ytest,1],xtest[ypred == ytest,7])
#plt.scatter(xtest[ypred != ytest,1],xtest[ypred != ytest,7])
#plt.title("Predicted and Missed Predicted")
#plt.legend(["Correctly Predicted","Incorrectly Predicted"])
#plt.xlabel('Glucose')
#plt.ylabel('Age')
#plt.show()
