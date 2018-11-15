import pandas as pd

# y = train.pop('Survived')
# X = train
def clean_data(df):
    df = df.drop('Cabin', axis=1)
    df['Embarked'] = df['Embarked'].fillna('S')
    for pclass in df['Pclass'].unique():
        df.loc[(df['Pclass'] == pclass) & (df['Age'].isnull())] = df.groupby('Pclass')['Age'].mean().loc[pclass]

    return df

if __name__ == '__main__':
    test = pd.read_csv('/Users/johnelmer/Desktop/test.csv')
    train = pd.read_csv('/Users/johnelmer/Desktop/train.csv')
    train = clean_data(train)
    test = clean_data(test)
    print(test.head())
    print(train.head())
