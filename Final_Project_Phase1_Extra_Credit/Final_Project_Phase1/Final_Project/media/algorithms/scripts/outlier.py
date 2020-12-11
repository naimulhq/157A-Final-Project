import csv
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import pathlib
from mpld3 import plugins
import plotly.express as px
import plotly.io as pio
import pickle
selected_features = ['TRB','AST','PTS']


def generating_scores(data, scores,modelName):
    sortedIndices = np.argsort(scores)
    sortedNames = data['Player'].iloc[sortedIndices]
    sortedScores = np.array(scores)[sortedIndices]
    sortedResults = zip(sortedNames,sortedScores)
    with open(modelName + '.csv','w',newline='',encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(('Name','Result'))
        for i in sortedResults:
            writer.writerow(i)
        

def scatter_with_outliers(data, outliers, topThree, outliersSansTopThree, title):
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(data.iloc[~outliers][selected_features[0]],
               data.iloc[~outliers][selected_features[1]],
               data.iloc[~outliers][selected_features[2]])
    ax.scatter(data.iloc[outliersSansTopThree][selected_features[0]],
               data.iloc[outliersSansTopThree][selected_features[1]],
               data.iloc[outliersSansTopThree][selected_features[2]])
    ax.scatter(topThree[selected_features[0]],
               topThree[selected_features[1]],
               topThree[selected_features[2]])
    plt.savefig(title + ' Scatter With Three Outliers.png')
    plt.close()


def top_three_outliers(data, topThree, title):
    fig, ax = plt.subplots(3, 3, sharex=True, squeeze=False)
    for rownum, (_,obj) in enumerate(topThree.iterrows()):
        for colnum, stat in enumerate(selected_features):
            hist, bin_edges = np.histogram(data[stat], bins=100)
            cdf = np.cumsum(hist)/data.shape[0]
            bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
            ax[rownum,colnum].plot(bin_centers,cdf)
            ax[rownum,colnum].set_title(obj["Player"] + ' on ' + stat)
            ax[rownum,colnum].plot([obj[stat], obj[stat]],[0.0,1.0])
    plt.savefig(title + ' Top Three Outliers.png')
    plt.close()


def load_data(path):
    data = pd.read_csv(path)
    return data.drop(['FG%','3P%','2P%','eFG%','FT%'],axis=1)
    

def load_clean_data(path):
    data = pd.read_csv(path)[['Player']+selected_features]
    for stat in selected_features:
        data[stat] = data[stat]/data[stat].max()
    return data


def plotDistributions(data):
    for start in data.columns[5:]:
        fig,ax = plt.subplots(1,2,sharex=True,squeeze=True)
        hist, bin_edges = np.histogram(data[start],bins=100)
        bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
        ax[0].bar(bin_centers,hist,width=np.diff(bin_edges))
        ax[0].set_title(label=start+" PDF")
      
        cdf = np.cumsum(hist)/data.shape[0]
        ax[1].plot(bin_centers,cdf)
        ax[1].set_title(label=start+" CDF")

        plt.savefig('Distribution of ' + start + '.png')
        plt.close()


def plotScatter(data):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(data[data.columns[23-5]],data[data.columns[24-5]],data[data.columns[29-5]])
    ax.set_xlabel(data.columns[23-5])
    ax.set_ylabel(data.columns[24-5])
    ax.set_zlabel(data.columns[29-5])
    #plt.show()
    plt.savefig('Scatter of TRB AST PTS.png')
    plt.close()

def train_one_class_svm(data, kernel, random_state):
    from sklearn.svm import OneClassSVM
    return OneClassSVM(kernel=kernel).fit(data[selected_features])

def train_elliptic_envelope(data, contamination, random_state):
    from sklearn.covariance import EllipticEnvelope
    return EllipticEnvelope(contamination=contamination, random_state=random_state).fit(data[selected_features])

def train_isolation_forest(data, contamination, random_state):
    from sklearn.ensemble import IsolationForest
    return IsolationForest(contamination=contamination,random_state=random_state).fit(data[selected_features])


# path = pathlib.Path(__file__).parents[1]

# # path = "C://Users//Edwin//OneDrive//Documents//UCSB//ECE 157A//Final_Project_Phase1_Extra_Credit//Final_Project_Phase1//Final_Project//media//algorithms//saved_models//nba_players_stats_19_20_per_game.csv"
# # data = load_data(path)
# #plotDistributions(data)
# # plotScatter(data)

# data = load_clean_data(path/'saved_models//nba_players_stats_19_20_per_game.csv')

# kernel = 'rbf'
# random_state = 42
# contamination = 0.05

# clfSVM = train_one_class_svm(data,kernel,random_state)
# # clfEE = train_elliptic_envelope(data,contamination,random_state)
# # clfIF =  train_isolation_forest(data,contamination,random_state)

# scoreSVM = clfSVM.decision_function(data[selected_features])
# # scoreEE = clfEE.decision_function(data[selected_features])
# # scoreIF = clfIF.decision_function(data[selected_features])

# topThreeIndexSVM = np.argsort(scoreSVM)[:3]
# topThreeSVM = data.iloc[topThreeIndexSVM]

# # topThreeIndexEE = np.argsort(scoreEE)[:3]
# # topThreeEE = data.iloc[topThreeIndexEE]

# # topThreeIndexIF = np.argsort(scoreIF)[:3]
# # topThreeIF = data.iloc[topThreeIndexIF]

# outliersSVM = scoreSVM < 0
# outliersSansTopThreeSVM = outliersSVM.copy()
# outliersSansTopThreeSVM[topThreeIndexSVM] = False

# labels = []

# for i in range(outliersSVM.shape[0]):
#     label = outliersSVM[i]
#     if label == True:
#         labels.append('inlier') 
#     else:
#         labels.append('outlier')
# labels = np.array(labels)
# labels[topThreeIndexSVM] = 'TopThree'
# outliersEE = scoreEE < 0
# outliersSansTopThreeEE = outliersEE.copy()
# outliersSansTopThreeEE[topThreeIndexEE] = False

# outliersIF = scoreIF < 0
# outliersSansTopThreeIF = outliersIF.copy()
# outliersSansTopThreeIF[topThreeIndexIF] = False

# top_three_outliers(data,topThreeSVM,'SVM')
# top_three_outliers(data,topThreeEE,'EE')
# top_three_outliers(data,topThreeIF,'IF')

# scatter_with_outliers(data, outliersSVM, topThreeSVM, outliersSansTopThreeSVM, 'SVM')
# scatter_with_outliers(data, outliersEE, topThreeEE, outliersSansTopThreeEE, 'EE')
# scatter_with_outliers(data, outliersIF, topThreeIF, outliersSansTopThreeIF, 'IF')

# generating_scores(data,scoreSVM,'SVMScore')
# generating_scores(data,scoreEE,'EEScore')
# generating_scores(data,scoreIF,'IFScore')


def run_algo1(file_path, debug=False):
    loaded_model = pickle.load(open("C://Users//Edwin//OneDrive//Documents//UCSB//ECE 157A//Final//Final_Project_Phase1_Extra_Credit//Final_Project_Phase1//Final_Project//media//algorithms//saved_models//outlier.sav", 'rb'))
    data = load_data(file_path)
    data = load_clean_data(file_path)
    scoreSVM = loaded_model.decision_function(data[selected_features])
    # scoreEE = clfEE.decision_function(data[selected_features])
    # scoreIF = clfIF.decision_function(data[selected_features])

    topThreeIndexSVM = np.argsort(scoreSVM)[:3]
    topThreeSVM = data.iloc[topThreeIndexSVM]

    # topThreeIndexEE = np.argsort(scoreEE)[:3]
    # topThreeEE = data.iloc[topThreeIndexEE]

    # topThreeIndexIF = np.argsort(scoreIF)[:3]
    # topThreeIF = data.iloc[topThreeIndexIF]

    outliersSVM = scoreSVM < 0
    outliersSansTopThreeSVM = outliersSVM.copy()
    outliersSansTopThreeSVM[topThreeIndexSVM] = False

    labels = []

    for i in range(outliersSVM.shape[0]):
        label = outliersSVM[i]
        if label == True:
            labels.append('inlier') 
        else:
            labels.append('outlier')
    labels = np.array(labels)
    labels[topThreeIndexSVM] = 'TopThree'



    # data = preprocess_data(colNames,data)
    # ypred = clfSVM.predict(data)
    # Load file_path instead of unknowns.csv
    # Do the same processing you did to
    # get your unknowns.csv results from your model.
    # Get the figure object Matplotlib will be working with:
    # def scatter_with_outliers(data, outliers, topThree, outliersSansTopThree, title):
    #                          (data, outliersSVM, topThreeSVM, outliersSansTopThreeSVM, 'SVM')
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection='3d')
    # ax.scatter(data.iloc[~outliersSVM][selected_features[0]],
    #            data.iloc[~outliersSVM][selected_features[1]],
    #            data.iloc[~outliersSVM][selected_features[2]])
    # ax.scatter(data.iloc[outliersSansTopThreeSVM][selected_features[0]],
    #            data.iloc[outliersSansTopThreeSVM][selected_features[1]],
    #            data.iloc[outliersSansTopThreeSVM][selected_features[2]])
    # ax.scatter(topThreeSVM[selected_features[0]],
    #            topThreeSVM[selected_features[1]],
    #            topThreeSVM[selected_features[2]])
  
    # fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
    #           color='species')
    data['labels'] = labels
    fig = px.scatter_3d(data,x=selected_features[0], y=selected_features[1], z = selected_features[2],color='labels')
                        # labels={'0': 'inlier', '1': "outlier",'2':"Top Three"})
    
    # fig.show()
    # fig['data'][0]['showlegend']=True
    
    fig = pio.to_html(fig)
    return fig
    # if debug:
    #     plt.show()
    # Scatterplot your results, as per
    # any of your scatterplots from homework 1.
    # Make sure to use plt.xlabel(), plt.ylabel()
    # and plt.title() to give clear labeling!
    # plugins.connect(fig, plugins.MousePosition(fontsize=12, fmt='.3g'))
    # plugins.connect(fig, plugins.Zoom(button=True, enabled=None))
    # with io.StringIO() as stringbuffer:     
    #     fig.savefig(stringbuffer,format='svg')
    #     svgstring = stringbuffer.getvalue()
    # return svgstring

    #plotly io function tohtml


    # return mpld3.fig_to_html(fig)


# if __name__=="__main__":
#     run_algo1("C:\\Users\\Edwin\\OneDrive\\Documents\\UCSB\\ECE 157A\\Final_Project_Phase1\\Final_Project_Phase1\\Final_Project\\media\\upload\\nba_players_stats_19_20_per_game.csv", False)

