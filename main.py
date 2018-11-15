import pandas as pd

# y = train.pop('Survived')
# X = train
def clean_data(df):
    df = df.drop('Cabin', axis=1)
    df = df.drop('Ticket', axis=1)
    df = df.drop('Name', axis=1)
    df = df.drop('PassengerId', axis=1)
    df['Sex'].replace({'male': 0, 'female': 1},inplace=True)
    df['Embarked'] = df['Embarked'].fillna('S')
    for pclass in df['Pclass'].unique():
        df.loc[(df['Pclass'] == pclass) & (df['Age'].isnull())] = df.groupby('Pclass')['Age'].mean().loc[pclass]

    return df

def one_hot_encode(df):
    df['Embarked_S'] =0
    df['Embarked_C'] =0
    df['Embarked_Q'] =0
    df.loc[df['Embarked'] == 'S', 'Embarked_S'] = 1
    df.loc[df['Embarked'] == 'C', 'Embarked_C'] = 1
    df.loc[df['Embarked'] == 'Q', 'Embarked_Q'] = 1
    df = df.drop('Embarked', axis=1)

    return df

def flag_infant(df):
    df['infant'] = df['Age'] < 10
    df = df.drop('Age', axis=1)

    return df

def extract_title(df):
    df['Name'].str.split(',').apply(lambda x: x[1]).str.split('.').apply(lambda x: x[0]).unique()

    return df

if __name__ == '__main__':
    test = pd.read_csv('/Users/johnelmer/Desktop/test.csv')
    train = pd.read_csv('/Users/johnelmer/Desktop/train.csv')
    train = clean_data(train)
    test = clean_data(test)
    train = one_hot_encode(train)
    print(test.head())
    print(train.head())
