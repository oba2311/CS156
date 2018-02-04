"""A clear report explaining
which variables you included in the model
any cleaning or transformations that you carried out on the data
the type of model you used and any settings that the model required
the training method you used, and any techniques that you used to avoid overfitting the data
an estimate of how well the model will perform on unseen data"""

"""The code for:
cleaning and transforming the data (if any)
fitting the model
evaluating its performance"""

"""
Data is from:  https://www.lendingclub.com/info/download-data.action
Relevant tutorials: https://www.youtube.com/watch?v=KM1dtuOkb4Y
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

LOAN_DATA = "/Users/oba2311/CS156/week 3/LoanStats3d_2015.csv"
REJECT_DATA = "/Users/oba2311/CS156/week 3/RejectStatsD_2015.csv"


def thousand_lines_generator(csv=REJECT_DATA):
    """
    Generates a spreadsheet with first 1000 rows.
    :param REJECT_DATA: csv with all the data
    :return: creates a csv with first 100 rows.
    """
    with open(csv) as loan_file:
        with open('Reject_2015_first_1000.csv', 'w') as out_file:
            for _ in range(1000):
                out_file.write(loan_file.readline())


def add_lables(loans="LoanStats_2015_first_1000.csv", reject="Reject_2015_first_1000.csv"):
    """
    the function takes the csvs and adds a label column to them.
    :param loans: (approved) loans csv.
    :param reject: rejected loans csv.
    :return: the two csvs with dummy 1s and 0s for approved and rejected column.
    """
    loans = pd.read_csv(loans)
    loans["accept/reject"] = (np.ones(len(loans)))
    rejected = pd.read_csv(reject)
    rejected["accept/reject"] = (np.zeros(len(rejected)))
    return loans, rejected


def merge_data(loans=add_lables()[0], rejected=add_lables()[1]):
    combined_data = pd.concat(
        [rejected, loans.rename(columns={'loan_amnt': 'Amount requested', "emp_length": "Employment Length",
                                         "dti": "Debt-to-income ratio"})])
    combined_data = combined_data.to_csv("small_combined_data.csv")
    return combined_data


def p2f(x):
    return float(x.strip('%')) / 100


def data_prep_for_model():
    f = pd.read_csv("small_combined_data.csv")
    features = f.drop("accept/reject", axis=1)
    features[] = features[]
    labels = f["accept/reject"]
    return features, labels


def data_train_test_splitter(X=data_prep_for_model()[0], y=data_prep_for_model()[1]):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(X=data_train_test_splitter()[0], y=data_train_test_splitter()[1]):
    model = LogisticRegression()
    try:
        model = model.fit(X, y)
    except ValueError as e:
        print(e)
    print(model.score(X, y))


train_model()
