# import libraries
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV

import re
import numpy as np
import pickle
import sys


def load_data(database_filepath):
    """
    Loads the data of the given file name and converts in X and Y variable.

    :param database_filepath: sql file name from where data is extracted.
    :return: input variables, target_variables, target_variable names
    """
    engine = create_engine('sqlite:///' + database_filepath )
    df = pd.read_sql_table('DisasterResponse', engine)
    cols = ['related',
            'request', 'offer', 'aid_related', 'medical_help', 'medical_products',
            'search_and_rescue', 'security', 'military', 'child_alone', 'water',
            'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees',
            'death', 'other_aid', 'infrastructure_related', 'transport',
            'buildings', 'electricity', 'tools', 'hospitals', 'shops',
            'aid_centers', 'other_infrastructure', 'weather_related', 'floods',
            'storm', 'fire', 'earthquake', 'cold', 'other_weather',
            'direct_report']
    X = df['message']
    Y = df[cols]

    return X,Y,cols


def tokenize(text):
    stop_words = list(stopwords.words("english"))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]"," ",text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [
        lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words
    ]
    return clean_tokens


def build_model():


    pipeline = Pipeline([
        ('vect' , CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer(sublinear_tf = False)),
        ('moc' , MultiOutputClassifier(estimator=RandomForestClassifier(
            n_jobs = 1,
            n_estimators = 100,
            random_state = 100,
            criterion = 'gini',
            max_depth=5,
            max_features=0.3,
            min_samples_split=4
        )))])

    parameters = {
        'moc__estimator__min_samples_split': (3, 4),
        'moc__estimator__max_features': ('sqrt', 0.3),
        'moc__estimator__max_depth': (3, 5),
        'moc__estimator__criterion': ('gini','entropy'),
    }


    cv = GridSearchCV(pipeline, param_grid=parameters,verbose= 10)

    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    """
    evaluates model on test set.
    :param model: learning model.
    :param X_test: Input test variable.
    :param y_test: Target test variable.
    :param category_names: column names of target variable.
    :return: score of model on test set : dataframe
    """
    y_pred = model.predict(X_test)
    accuracy = [[(y_pred[:, i] == y_test.values[:, i]).mean(),
                 *precision_recall_fscore_support(
                     y_test.values[:, i],
                     y_pred[:, i],
                     average='weighted',
                     labels=np.unique(y_pred[:, i]))]
                for i in range(y_pred.shape[1])]
    accuracy = np.array(accuracy)[:, :-1]
    accuracy = (accuracy * 10000).astype(int) / 100

    print('Showing scores...')
    print('\nAverage scores for all indicators...')
    scores = pd.DataFrame(
        data=accuracy,
        index=category_names,
        columns=['Accuracy %', 'Precision %', 'Recall %', 'F-score %'])
    print(scores.mean(axis=0))
    print('\Detailed scores for each indicator...')
    print(scores)
    pass



def save_model(model, model_filepath):
    """
    Saves the model in pickle file format.
    :param model: model to be saved.
    :param model_filepath: name of the file.
    :return: None
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
