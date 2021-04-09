import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def clean_data(df):
    df = df.drop('Cabin', axis=1)
    df = df.drop('Ticket', axis=1)
    df = df.drop('Name', axis=1)
    df = df.drop('PassengerId', axis=1)
    df['Sex'].replace({'male': 0, 'female': 1},inplace=True)
    df['Embarked'] = df['Embarked'].fillna('S')
    df = impute_by_class(df, 'Age')
    df = impute_by_class(df, 'Fare')

    return df

def impute_by_class(df, field):
    for pclass in df['Pclass'].unique():
        df.loc[(df['Pclass'] == pclass) & (df[field].isnull())] = df.groupby('Pclass')[field].mean().loc[pclass]

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


def build_model():
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(penalty='l1', C=1000))
    ])
    return model

if __name__ == '__main__':
    train = pd.read_csv('./data/train.csv')
    # print(train['Survived'])
    y = train.pop('Survived')
    train = clean_data(train)
    train = one_hot_encode(train)


    model = build_model()
    # from sklearn.preprocessing import LabelEncoder
    # lab_enc = LabelEncoder()
    # tscores_encoded = lab_enc.fit_transform(y)

    # gs = GridSearchCV(
    #     model,
    #     {'logreg__penalty': ['l1','l2'],
    #     'logreg__C': [1000, 100, 1, 10]},
    #     cv=5,
    #     n_jobs=4
    # )
    # print(train.isnull().sum())
    model.fit(train, y)
    # print(train.head())
    test = pd.read_csv('./data/test.csv')
    test = clean_data(test)
    test = one_hot_encode(test)
    # print(test.head())
    prediction = model.predict(test)

    pd.DataFrame({'PassengerId': test.index.values + 892, 'Survived': prediction}).to_csv('results.csv', index = False)
