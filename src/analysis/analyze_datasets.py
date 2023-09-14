import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src import load_dataset as ld
from src import bias_techniques as bt
from sklearn.preprocessing import MinMaxScaler

def feature_distribution(dataset_name):
    X, y, feature_names = ld.load_breast_cancer(True)
    X = MinMaxScaler().fit_transform(X)
    #selected_ids = bt.bias_select_by_feature(X, feature_id=0)
    selected_ids = bt.bias_by_dirichlet(X)
    df_num = pd.DataFrame(X[selected_ids,:5], columns=feature_names[:5])
    df_num['class'] = y[selected_ids]
    #df_num = pd.DataFrame(X[:,:5], columns=feature_names[:5])
    #df_num['class'] = y
    sns.pairplot(df_num, hue='class')
    #df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
    plt.show()

feature_distribution('breast_cancer')