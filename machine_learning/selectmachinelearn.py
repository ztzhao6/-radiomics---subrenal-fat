# select model & select the parameters for the specified model

import pandas as pd
import numpy as np
import os
import pymrmr
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from sklearn.feature_selection import SelectFromModel


class MRMRProcess(BaseEstimator, TransformerMixin):
    def __init__(self, selection_method='MIQ', selected_num=200):
        self.selection_method = selection_method
        self.selected_num = selected_num

    def get_params(self, deep=True):
        return {"selection_method": self.selection_method, "selected_num": self.selected_num}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        X_frame = pd.DataFrame(X, index=list(y.index), columns=[str(i) for i in range(X.shape[1])])
        self.selected_mask = []
        data_frame = pd.concat([y, X_frame], axis=1)
        all_features = X_frame.columns.tolist()
        selected_features = pymrmr.mRMR(data_frame, self.selection_method, self.selected_num)
        for i in range(len(all_features)):
            if all_features[i] in selected_features:
                self.selected_mask.append(True)
            else:
                self.selected_mask.append(False)
        return self

    def transform(self, X, y=None):
        X_frame = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
        selected_features = []
        all_features = X_frame.columns.tolist()
        for i in range(len(self.selected_mask)):
            if self.selected_mask[i]:
                selected_features.append(all_features[i])
        X_transformed = X_frame[selected_features]
        return X_transformed

    def get_mask(self):
        return self.selected_mask


def get_results(train_path, save_result_path, save_model_path=None):
    data_frame = pd.read_csv(train_path)
    X = data_frame.drop(['Image', 'label'], axis=1, inplace=False)
    y = data_frame['label']

    # select model
    # pipe = Pipeline([
    #     ('scaler', None),
    #     ('univariate_select', SelectKBest(k=500)),
    #     # ('mrmr', MRMRProcess()),
    #     ('feature_select', None),
    #     ('classify', None)
    # ])
    #
    # param_grid = {
    #     'scaler': [StandardScaler(), MinMaxScaler()],
    #     'feature_select':
    #         [RFECV(LinearSVC(C=0.5, penalty="l1", dual=False, max_iter=100000), step=6, cv=StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0), scoring='roc_auc'),
    #          RFECV(ExtraTreesClassifier(n_estimators=100, random_state=0), step=6, cv=StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0), scoring='roc_auc')],
    #     'classify': [LogisticRegression(C=1.0, solver='saga', max_iter=100000),
    #                  LinearSVC(C=0.5, penalty="l1", dual=False, max_iter=100000),
    #                  ExtraTreesClassifier(n_estimators=100, random_state=0),
    #                  GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=1, random_state=0)]
    # }

    # select the parameters for the specified model
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('univariate_select', SelectKBest(k=500)),
        ('feature_select', SelectFromModel(LinearSVC(penalty="l1", dual=False, max_iter=100000))),
        # ('feature_select', RFECV(LinearSVC(penalty="l1", dual=False, max_iter=100000), step=1, cv=StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0), scoring='roc_auc')),
        ('classify', LinearSVC(penalty="l1", dual=False, max_iter=100000))
    ])

    param_grid = {
        'univariate_select__k': [300, 500, 700],
        'feature_select__estimator__C': [2],
        'classify__C': [0.1]
    }

    # param_grid = [
    #     {
    #         'scaler': [StandardScaler(), MinMaxScaler()],
    #         'feature_select': [RFECV(LinearSVC(C=0.5, penalty="l1", dual=False, max_iter=10000), step=6, cv=StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0), scoring='roc_auc')],
    #         'classify': [LogisticRegression(solver='saga', max_iter=10000)],
    #         'feature_select__estimator__C': [0.01, 0.1, 1, 10, 100],
    #         'classify__C': [0.01, 0.1, 1, 10, 100]
    #     },
    #     {
    #         'scaler': [StandardScaler(), MinMaxScaler()],
    #         'feature_select': [RFECV(LinearSVC(C=0.5, penalty="l1", dual=False, max_iter=10000), step=6, cv=5, scoring='roc_auc')],
    #         'classify': [LinearSVC(C=0.5, penalty="l1", dual=False, max_iter=10000)],
    #         'feature_select__estimator__C': [0.01, 0.1, 1, 10, 100],
    #         'classify__C': [0.01, 0.1, 1, 10, 100]
    #     },
    #     {
    #         'scaler': [StandardScaler(), MinMaxScaler()],
    #         'feature_select': [RFECV(LinearSVC(C=0.5, penalty="l1", dual=False, max_iter=10000), step=6, cv=5, scoring='roc_auc')],
    #         'classify': [ExtraTreesClassifier(n_estimators=100, random_state=0)],
    #         'feature_select__estimator__C': [0.01, 0.1, 1, 10, 100]
    #     },
    #     {
    #         'scaler': [StandardScaler(), MinMaxScaler()],
    #         'feature_select': [RFECV(LinearSVC(C=0.5, penalty="l1", dual=False, max_iter=10000), step=6, cv=5, scoring='roc_auc')],
    #         'classify': [GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=1, random_state=0)],
    #         'feature_select__estimator__C': [0.01, 0.1, 1, 10, 100],
    #         'classify__max_depth': [1, 3],
    #         'classify__max_leaf_nodes': [2, 10, None],
    #         'classify__learning_rate': [0.1, 0.5, 1]
    #     },
    #     {
    #         'scaler': [StandardScaler(), MinMaxScaler()],
    #         'feature_select': [RFECV(ExtraTreesClassifier(n_estimators=100, random_state=0), step=6, cv=5, scoring='roc_auc')],
    #         'classify': [LogisticRegression(solver='saga', max_iter=1000)],
    #         'classify__C': [0.01, 0.1, 1, 10, 100]
    #     },
    #     {
    #         'scaler': [StandardScaler(), MinMaxScaler()],
    #         'feature_select': [
    #             RFECV(ExtraTreesClassifier(n_estimators=100, random_state=0), step=6, cv=5, scoring='roc_auc')],
    #         'classify': [LinearSVC(C=0.5, penalty="l1", dual=False, max_iter=10000)],
    #         'classify__C': [0.01, 0.1, 1, 10, 100]
    #     },
    #     {
    #         'scaler': [StandardScaler(), MinMaxScaler()],
    #         'feature_select': [
    #             RFECV(ExtraTreesClassifier(n_estimators=100, random_state=0), step=6, cv=5, scoring='roc_auc')],
    #         'classify': [ExtraTreesClassifier(n_estimators=100, random_state=0)]
    #     },
    #     {
    #         'scaler': [StandardScaler(), MinMaxScaler()],
    #         'feature_select': [
    #             RFECV(ExtraTreesClassifier(n_estimators=100, random_state=0), step=6, cv=5, scoring='roc_auc')],
    #         'classify': [GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=1, random_state=0)],
    #         'classify__max_depth': [1, 3],
    #         'classify__max_leaf_nodes': [2, 10, None],
    #         'classify__learning_rate': [0.1, 0.5, 1]
    #     },
    # ]

    grid = GridSearchCV(pipe, cv=StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0),
                        n_jobs=-3, param_grid=param_grid, iid=False, scoring='roc_auc', refit=True)

    grid.fit(X, y)

    results = pd.DataFrame(grid.cv_results_)
    results.to_csv(save_result_path, index=False)

    if save_model_path:
        joblib.dump(grid.best_estimator_, save_model_path)


if __name__ == '__main__':
    # select model
    # train_path = 'E:/hypertension/process&result/process_features/train/train_20.CSV'
    # save_result_path = 'E:/hypertension/process&result/results/select_model/result.CSV'
    #
    # get_results(train_path, save_result_path)

    # select the parameters for the specified model
    train_root = 'E:/hypertension/process&result/results_1.0/process_features/train/'
    save_result_root = 'E:/hypertension/process&result/results_1.0/results/train_result/'
    save_model_root = 'E:/hypertension/process&result/results_1.0/results/train_result/'
    for train_path_name in os.listdir(train_root):
        if train_path_name == 'train_4.CSV':
            train_path = train_root + train_path_name
            save_result_path = save_result_root + train_path_name.split('.')[0] + '_result.CSV'
            save_model_path = save_model_root + train_path_name.split('.')[0] + '_model.pkl'
            get_results(train_path, save_result_path, save_model_path)


