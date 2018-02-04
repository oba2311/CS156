# Split your dataset from the PCA pre-class work into 80% training data and 20% testing data.
import Data_prep_for_LDA
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def prepare_data():
    X_train, X_test, y_train, y_test = train_test_split(Data_prep_for_LDA.mini_main()[0],
                                                        Data_prep_for_LDA.mini_main()[1],
                                                        test_size=0.2)
    return X_train, X_test, y_train, y_test


"""
Build a simple linear classifier using the original pixel data. There are several options that you can try including 
a linear SVC (http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#examples-using-sklearn-svm-linearsvc) 
"""


def linear_SVC_clf(X=prepare_data()[0], y=prepare_data()[1]):
    clf = LinearSVC(random_state=0)
    clf.fit(X, y)
    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=1000,
              multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
              verbose=0)
    return clf

"""
or 
a logistic classifier (http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#examples-using-sklearn-linear-model-logisticregression) 
both of which will be covered in more detail later in this course. 
What is your error rate on the training data? What is your error rate on your testing data?

Train the same linear model as in question 1, but now on the reduced representation that you created using PCA.
What is your error rate on the training data? What is your error rate on your testing data?
Train the same linear model as in question 1, but now on the reduced representation that you created using LDA. 
What is your error rate on the training data? What is your error rate on your testing data?

Write three paragraphs, describing and interpreting your results from questions 1, 2, and 3. 
Make a recommendation on which classifier you would prefer, and why.
"""
