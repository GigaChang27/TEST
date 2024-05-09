import pandas as pd

# import P30_DBQuery

DBPath='data/FinIndex/'

FinDB=pd.read_csv(DBPath+'FinDB_Rtn.csv',encoding='UTF-8')

FinDB['stockID']=FinDB['stockID'].astype(str)

FinDB=FinDB.set_index(keys=['stockID','yyyyQ'])





FinDB.columns

features = ['權益報酬率', '營業毛利年增率','return']

featuresNoReturn = ['權益報酬率', '營業毛利年增率']

Dataset = FinDB[features].dropna(how='any')

Dataset.head()





Dataset.plot.scatter(features[0], features[1])



def is_valid(feature, nstd):

    ub = feature.mean() + nstd * feature.std()

    lb = feature.mean() - nstd * feature.std()



    return (feature > lb) & (feature <ub)



#valid = is_valid(Dataset['權益報酬率'], 2) & is_valid(Dataset['營業毛利年增率'], 0.05)

valid = is_valid(Dataset['權益報酬率'], 2) & is_valid(Dataset['營業毛利年增率'], 2)

Dataset = Dataset[valid].dropna()



Dataset['權益報酬率'].hist(bins=100)



import pandas as pd

import sklearn.preprocessing as preprocessing



Dataset_scaled = pd.DataFrame(preprocessing.scale(Dataset), index=Dataset.index, columns=Dataset.columns)

Dataset_scaled.head()



Dataset_scaled['權益報酬率'].hist(bins=100)

Dataset_scaled['營業毛利年增率'].hist(bins=100, alpha=0.5)

Dataset_scaled['return'] = Dataset['return']

# from sklearn.model_selection import train_test_split



# Dataset_train, Dataset_test = train_test_split(Dataset_scaled, test_size=0.1, random_state=0)



from sklearn.svm import SVC



cf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,

  decision_function_shape='ovr', degree=3, gamma='auto',

  kernel='rbf', max_iter=-1, probability=False, random_state=None,

  shrinking=True, tol=0.001, verbose=False)



cf.fit(Dataset_scaled[featuresNoReturn], Dataset_scaled['return'] > Dataset_scaled['return'].quantile(0.5))



from mlxtend.plotting import plot_decision_regions



# features_plot = Dataset_scaled[features].values

features_plot = Dataset_scaled[featuresNoReturn].values

labels_plot = (Dataset_scaled['return'] > Dataset_scaled['return'].quantile(0.5)).astype(int).values



plot_decision_regions(features_plot, labels_plot, cf)



# 回測



Dataset_scaled['predict'] = cf.predict(Dataset_scaled[featuresNoReturn])

Dataset_scaled = Dataset_scaled.reset_index()



dates = sorted(list(set(Dataset_scaled['yyyyQ'])))



Q_returns1 = []

Q_returns2 = []

for date in dates:

    current_stocks = Dataset_scaled[Dataset_scaled['yyyyQ'] == date]

    buy_stocks = current_stocks[current_stocks['predict'] == True]

    sell_stocks = current_stocks[current_stocks['predict'] == False]

    

    Q_return1 = buy_stocks['return'].mean()

    Q_returns1.append(Q_return1)

    

    Q_return2 = sell_stocks['return'].mean()

    Q_returns2.append(Q_return2)



import matplotlib.pyplot as plt

plt.style.use("ggplot")



pd.Series(Q_returns1, index=dates).cumprod().plot(color='red')

pd.Series(Q_returns2, index=dates).cumprod().plot(color='blue')
