import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

np.random.seed(7)

# for P test
def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)
    regressor_OLS.summary()
    return x, columns

# load df
df1 = pd.read_csv('Top_songs.csv')
df1 = df1.loc[:, ~df1.columns.str.contains('^Unnamed')]
df1.insert(0, 'Viral', np.full(len(df1), 1))
df2 = pd.read_csv('Random_songs.csv')
df2 = df2.loc[:, ~df2.columns.str.contains('^Unnamed')]
df2.insert(0, 'Viral', np.full(len(df2), 0))

# merge df and drop duplicating songs
df = pd.concat([df1, df2])
df = df.drop_duplicates(subset=['title'], keep = 'first')
df = df.drop(['id', 'title', 'all_artists'], axis=1)

# data normalization
#df=(df-df.mean())/df.std() # Z-Score Normalization
df=(df-df.min())/(df.max()-df.min())

# get heatmap
corr = df.corr()
ax = sns.heatmap(corr)
#plt.show()

# drop features with high correlation, >=0.8
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if abs(corr.iloc[i,j]) >= 0.8:
            if columns[j]:
                columns[j] = False
selected_columns = df.columns[columns]
df = df[selected_columns]

# calculate p value, considers p < 0.05 as significant
selected_columns = selected_columns[1:].values
#data_modeled, selected_columns = backwardElimination(df.iloc[:,1:].values, df.iloc[:,0].values, 0.8, selected_columns)

result = pd.DataFrame()
result['Viral'] = df.iloc[:,0]

#df = pd.DataFrame(data = data_modeled, columns = selected_columns)

#print(data)

'''
fig = plt.figure(figsize = (20, 20))
j = 0
for i in df.columns:
    plt.subplot(6, 4, j+1)
    j += 1
    sns.distplot(df[i][result['Viral']==0], color='g', label = 'flop')
    sns.distplot(df[i][result['Viral']==1], color='r', label = 'hit')
    plt.legend(loc='best')
fig.suptitle('Tik Tok Virality Analysis')
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()
'''
# split into test and train data
train, test = train_test_split(df, test_size = 0.2)
train.to_csv('train.csv')
test.to_csv('test.csv')