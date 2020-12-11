from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix, accuracy_score,explained_variance_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import mpld3
import csv
from sklearn.neighbors import KNeighborsClassifier
#import pandas as pd
import pathlib
from mpld3 import plugins
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
import sys, csv, pickle, plotly, plotly.express as px, plotly.graph_objects as go


def load_data(filename):
    with open(filename,'r') as infile:  # Reads the file
        reader = csv.reader(infile) # Used to grab labels
        colNames = next(reader) # Iterates and store labels
        rows = np.array(list(reader), dtype=np.float64) # Store Data and Cast to float64
    return colNames, rows

def seperate_labels(colNames, rows):
    labelCol = colNames.index('quality')    # label index
    ys = rows[:,labelCol]   # outcomes
    xs = np.delete(rows,labelCol, axis=1)   # deleting outcome column from data
    del colNames[labelCol]  # Removes outcome from label names
    return colNames, xs, ys

def train_test_split2(xs,ys,test_size=0.25,random_state=42):
    #xtrain0, xtest0, ytrain0, ytest0 = train_test_split(xs[ys==0],ys[ys==0],test_size=test_size,random_state=random_state)
    #xtrain1, xtest1, ytrain1, ytest1 = train_test_split(xs[ys==1],ys[ys==1],test_size=test_size,random_state=random_state)
    #xtrain2, xtest2, ytrain2, ytest2 = train_test_split(xs[ys==2],ys[ys==2],test_size=test_size,random_state=random_state)
    xtrain3, xtest3, ytrain3, ytest3 = train_test_split(xs[ys==3],ys[ys==3],test_size=test_size,random_state=random_state)
    xtrain4, xtest4, ytrain4, ytest4 = train_test_split(xs[ys==4],ys[ys==4],test_size=test_size,random_state=random_state)
    xtrain5, xtest5, ytrain5, ytest5 = train_test_split(xs[ys==5],ys[ys==5],test_size=test_size,random_state=random_state)
    xtrain6, xtest6, ytrain6, ytest6 = train_test_split(xs[ys==6],ys[ys==6],test_size=test_size,random_state=random_state)
    xtrain7, xtest7, ytrain7, ytest7 = train_test_split(xs[ys==7],ys[ys==7],test_size=test_size,random_state=random_state)
    xtrain8, xtest8, ytrain8, ytest8 = train_test_split(xs[ys==8],ys[ys==8],test_size=test_size,random_state=random_state)
    xtrain9, xtest9, ytrain9, ytest9 = train_test_split(xs[ys==9],ys[ys==9],test_size=test_size,random_state=random_state)
    #xtrain10, xtest10, ytrain10, ytest10 = train_test_split(xs[ys==10],ys[ys==10],test_size=test_size,random_state=random_state)

    xtrain = np.concatenate([xtrain3,xtrain4,xtrain5,xtrain6,xtrain7,xtrain8,xtrain9])
    xtest = np.concatenate([xtest3,xtest4,xtest5,xtest6,xtest7,xtest8,xtest9])
    ytrain = np.concatenate([ytrain3,ytrain4,ytrain5,ytrain6,ytrain7,ytrain8,ytrain9])
    ytest = np.concatenate([ytest3,ytest4,ytest5,ytest6,ytest7,ytest8,ytest9])

    return xtrain,xtest,ytrain,ytest
    
    
def preprocess_data(colNames, xs):
    bloodGlucoseIndex = 2
    newfeature = (xs[:,bloodGlucoseIndex]<1.0).reshape((xs.shape[0],1))
    xs = np.concatenate((xs,newfeature),axis=1)
    colNames.append('NoBloodGlucoseData')
    return colNames,xs

def write_data(filename, ypred):
   with open(filename,'w') as outfile:  # Reads the file
        writer = csv.writer(outfile, lineterminator = '\n') # Used to grab labels
        writer.writerow(['quality'])
        for i in ypred:
            writer.writerow([i])


def measurePerformance(clf,X_test,Y_test):
    scores = cross_val_score(clf, X_test, Y_test,cv=2)
    print("cross_val_score Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores


def hyperparameter(x_train, x_test, y_train, y_test):
    training_acc = []
    testing_acc = []
    x = range(1,20)
    for i in x:
        classifier = KNeighborsClassifier(n_neighbors=i).fit(x_train, y_train)
        train_acc = classifier.score(x_train, y_train)
        test_acc = classifier.score(x_test, y_test)
        training_acc.append(train_acc)
        testing_acc.append(test_acc)
        print("Training_Accuracy:\t", train_acc,"\tTest_Accuracy\t", test_acc)
    plt.plot(x, training_acc, label='training accuracy')
    plt.plot(x, testing_acc, label='test accuracy')
    plt.title("KNeighborsClassifier accuracy with n_neighbors")
    plt.xlabel('n_neighbors')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    return


def getCorrelationMatrix(data,names):
    a_data = dict()
    for i in range(len(names)):
        a_data[names[i]] = data[:,i]
    df = pd.DataFrame(a_data,columns = names)
    correlation = df.corr();
    return correlation


# Creates a scatter plot between features and quality
def featureQualityCorr(data,prediciton,f1,name):
    plt.scatter(prediciton,data[:,f1])
    plt.xlabel("Quality",fontsize=16)
    plt.ylabel(name,fontsize=16)
    plt.title("Quality vs " + name,fontsize=16)
    plt.show()



# random_state=40
# test_size=0.33
# path = pathlib.Path(__file__).parents[1]
# colNames, data = load_data(path/'saved_models//white_wine.csv')
# colNames, xs, ys = seperate_labels(colNames,data)

# ################################################## 1) ##################################################
# # plt.hist(ys)
# # plt.title("Histogram of Data")
# # plt.xlabel("Quality"), plt.ylabel("Quantity")
# # plt.show()

# # h = np.zeros((10))
# # for i in range(np.size(ys)):
# #     h[ ys[i].astype(int)  ] += 1

# # print("Quality\tQuanity")
# # for i in range(np.size(h)):
# #     print(i, "\t", h[i])

    
# ################################################## 2) ##################################################

# # colormap = plt.cm.cividis
# # correlation = getCorrelationMatrix(data,colNames)
# # sns.heatmap(correlation, cmap=colormap,annot = True)
# # plt.title("correlation of features")
# # plt.xlabel("features"), plt.ylabel("features")
# # plt.show()


# ################################################## 3) ##################################################

# #feature = 0
# #for i in range(11):
# #    featureQualityCorr(xs,ys,feature,colNames[feature])
# #    feature +=1
    

# ################################################## 6) ##################################################

# xtrain,xtest,ytrain,ytest = train_test_split2(xs,ys,test_size,random_state)

# # clfLR = LogisticRegression(random_state=random_state,dual=False,C=.2,penalty='l2',solver='newton-cg',max_iter=1000).fit(xtrain,ytrain)
# # ypred = np.round(clfLR.predict(xtest))
# # print("\nLogisticRegression")
# # print("Explained_Variance:\t",explained_variance_score(ytest,ypred))
# # print("CLF Score:\t\t",clfLR.score(xtest,ytest))
# # print("Accuracy Score:\t\t", accuracy_score(ytest,ypred))
# # scoreLR = measurePerformance(clfLR,xtest,ytest)
# # print("Confusion Matrix:\n",confusion_matrix(ytest,ypred))
# # colNames, data = load_data('unknowns.csv')
# # ypred = clfLR.predict(data)
# # write_data('scoreLR.csv', ypred)

# # clfMLP = MLPRegressor(hidden_layer_sizes=(150,2),activation='relu', solver='adam',alpha=0.001, max_iter=1000,learning_rate='constant',
# #                    learning_rate_init=0.001,random_state=random_state).fit(xtrain,ytrain)
# # ypred = np.round(clfMLP.predict(xtest))
# # print("\nMLPregressor")
# # print("Explained_Variance:\t",explained_variance_score(ytest,ypred))
# # print("CLF Score:\t\t",clfMLP.score(xtest,ytest))
# # print("Accuracy Score:\t\t",accuracy_score(ytest,ypred))
# # scoreMLP = measurePerformance(clfMLP,xtest,ytest)
# # print("Confusion Matrix:\n",confusion_matrix(ytest,ypred))
# # colNames, data = load_data('unknowns.csv')
# # ypred = clfLR.predict(data)
# # write_data('scoreMLP.csv', ypred)


# kernel = 1.0*RBF(1.0) + DotProduct() + WhiteKernel()

# # clfSVC = make_pipeline(StandardScaler(), SVC(C=1.5, gamma=1.5, kernel=kernel)).fit(xtrain,ytrain)
# # #clfSVC = svc(c=1.5, gamma=1.5, kernel='rbf', random_state=random_state).fit(xtrain,ytrain)
# # ypred = np.round(clfSVC.predict(xtest))
# # print("\nSVC")
# # print("Explained_Variance:\t",explained_variance_score(ytest,ypred))
# # print("CLF Score:\t\t",clfSVC.score(xtest,ytest))
# # print("Accuracy Score:\t\t",accuracy_score(ytest,ypred))
# # scoreSVC = measurePerformance(clfSVC,xtest,ytest)
# # print("Confusion Matrix:\n",confusion_matrix(ytest,ypred))
# # colNames, data = load_data('unknowns.csv')
# # ypred = clfSVC.predict(data)
# # write_data('scoreSVC.csv', ypred)

# clfSVR = make_pipeline(StandardScaler(), SVR(C=1.5, gamma=1.5, kernel=kernel)).fit(xtrain,ytrain)
# # clfSVR = svr(c=1.5, gamma=1.5, kernel='rbf').fit(xtrain,ytrain)
# #ypred = np.round(clfSVR.predict(xtest))
# # print("\nSVR")
# # print("Explained_Variance:\t",explained_variance_score(ytest,ypred))
# # print("CLF Score:\t\t",clfSVR.score(xtest,ytest))
# # print("Accuracy Score:\t\t",accuracy_score(ytest,ypred))
# # scoreSVR = measurePerformance(clfSVR,xtest,ytest)
# # print("Confusion Matrix:\n",confusion_matrix(ytest,ypred))
# # colNames, data = load_data('unknowns.csv')
# # ypred = clfSVR.predict(data)
# # write_data('scoreSVR.csv', ypred)

# # clfGPC = GaussianProcessClassifier(kernel=kernel,max_iter_predict=100,random_state=random_state).fit(xtrain,ytrain)
# # #clfGPC = gaussianprocessclassifier(random_state=random_state).fit(xtrain,ytrain)
# # ypred = np.round(clfGPC.predict(xtest))
# # print("\nGPC")
# # print("Explained_Variance:\t",explained_variance_score(ytest,ypred))
# # print("Clf Score:\t\t",clfGPC.score(xtest,ytest))
# # print("Accuracy Score:\t\t",accuracy_score(ytest,ypred))
# # scoreGPC = measurePerformance(clfGPC,xtest,ytest)
# # print("Confusion Matrix:\n",confusion_matrix(ytest,ypred))
# # colNames, data = load_data('unknowns.csv')
# # ypred = clfGPC.predict(data)
# # write_data('scoreGPC.csv', ypred)

# # kernel = 1.0*RBF(1.0) + DotProduct() + WhiteKernel()
# # clfGPR = GaussianProcessRegressor(kernel=kernel,random_state=random_state).fit(xtrain,ytrain)
# # #clfGPR = gaussianprocessregressor(random_state=random_state).fit(xtrain,ytrain)
# # ypred = np.round(clfGPR.predict(xtest))
# # print("\nGPR")
# # print("Explained_Variance:\t",explained_variance_score(ytest,ypred))
# # print("CLF Score:\t\t",clfGPR.score(xtest,ytest))
# # print("Accuracy Score:\t\t",accuracy_score(ytest,ypred))
# # scoreGPR = measurePerformance(clfGPR,xtest,ytest)
# # print("Confusion Matrix:\n",confusion_matrix(ytest,ypred))
# # colNames, data = load_data('unknowns.csv')
# # ypred = clfGPR.predict(data)
# # write_data('scoreGPR.csv', ypred)

# # print("\nTraining Validation")
# # print("LR")
# # measurePerformance(clfLR,xtrain,ytrain)
# # print("MLP")
# # measurePerformance(clfMLP,xtrain,ytrain)
# # print("SVC")
# # measurePerformance(clfSVC,xtrain,ytrain)
# # print("SVR")
# # measurePerformance(clfSVR,xtrain,ytrain)
# # print("GPC")
# # measurePerformance(clfGPC,xtrain,ytrain)
# # print("GPR")
# # measurePerformance(clfGPR,xtrain,ytrain)


# ################################################## 7) ##################################################

# ypred = clfSVR.predict(xtest).astype(int)



# def run_algo1(file_path, algo_model):
    # print("\n\n\n\nHereERERERERR ",file_path)
    # data = pd.read_csv(file_path)
    # model = pickle.load(open(algo_model,"rb"))
    # pred_y2= model.predict(data)
    # data.insert(11, "quality", pred_y2) 

    # fig = px.scatter(data, x= 'pH', y='alcohol', color='quality')

    # fig.update_layout(
    #     title="Logistic Regression of Data",
    #     xaxis_title="Alcohol",
    #     yaxis_title="pH",
    # )

    # return plotly.offline.plot(fig, output_type='div')




def run_algo1(file_path, debug=False):
    loaded_model = pickle.load(open("C://Users//Edwin//OneDrive//Documents//UCSB//ECE 157A//Final//Final_Project_Phase1_Extra_Credit//Final_Project_Phase1//Final_Project//media//algorithms//saved_models//regression.sav", 'rb'))
    colNames, data = load_data(file_path)
    

    ypred = np.round(loaded_model.predict(data))
    
    # Load file_path instead of unknowns.csv
    # Do the same processing you did to
    # get your unknowns.csv results from your model.
    # Get the figure object Matplotlib will be working with:
    


    fig = plt.figure()
    for i in range(3,10):           
        plt.scatter(data[ypred == i,7],data[ypred == i,10])

    # plt.scatter(data[ypred != ytest,7],data[ypred != ytest,10])
    plt.title("Predicted and Missed Predicted")
    plt.legend([i for i in range(3,10)])
    plt.xlabel('Density')
    plt.ylabel('Alcohol')

    if debug:
        plt.show()


    # Scatterplot your results, as per
    # any of your scatterplots from homework 1.
    # Make sure to use plt.xlabel(), plt.ylabel()
    # and plt.title() to give clear labeling!
    plugins.connect(fig, plugins.MousePosition(fontsize=12, fmt='.3g'))
    plugins.connect(fig, plugins.Zoom(button=True, enabled=None))

    return mpld3.fig_to_html(fig)

# if __name__=="__main__":
#     run_algo1("C:\\Users\\Edwin\\OneDrive\\Documents\\UCSB\ECE 157A\\Final_Project_Phase1\\Final_Project_Phase1\\Final_Project\\media\\upload\\unknowns_WgowrQ5.csv",False)


# threeMiss = 0;
# fourMiss = 0;
# fiveMiss = 0;
# sixMiss = 0;
# sevenMiss = 0;
# eightMiss = 0;
# nineMiss = 0;
# for i in range(len(ytest)):
#     if(ytest[i] == 3):
#         if(ytest[i] != ypred[i]):
#             threeMiss += 1
#     elif(ytest[i] == 4):
#         if(ytest[i] != ypred[i]):
#             fourMiss += 1
#     elif(ytest[i] == 5):
#         if(ytest[i] != ypred[i]):
#             fiveMiss += 1
#     elif(ytest[i] == 6):
#         if(ytest[i] != ypred[i]):
#             sixMiss += 1
#     elif(ytest[i] == 7):
#         if(ytest[i] != ypred[i]):
#             sevenMiss += 1
#     elif(ytest[i] == 8):
#         if(ytest[i] != ypred[i]):
#             eightMiss += 1
#     else:
#         if(ytest[i] != ypred[i]):
#             nineMiss += 1
# print("\nTotel missed predicted for Qualites:")
# print("3: ",threeMiss)
# print("4: ",fourMiss)
# print("5: ",fiveMiss)
# print("6: ",sixMiss)
# print("7: ",sevenMiss)
# print("8: ",eightMiss)
# print("9: ",nineMiss)

# total = threeMiss + fourMiss + fiveMiss + sixMiss + sevenMiss + eightMiss + nineMiss
# print("Totel Misses: ", total)