import pandas as pd

train = pd.read_csv('/Users/johnelmer/Desktop/train.csv')
y = train.pop('Survived')
# X = train
# test = pd.read_csv('/Users/johnelmer/Desktop/test.csv')


train.drop('Cabin', axis=1, inplace=True)
train['Embarked'] = train['Embarked'].fillna('S')
for pclass in train['Pclass'].unique():
    train.loc[train['Pclass'] == pclass, 'Age'] = train.groupby('Pclass')['Age'].mean().loc[pclass]

print(train.head())
