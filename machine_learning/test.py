import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def get_results(train_path, test_path, test_path_2, model_path, save_result_root):
    train_frame = pd.read_csv(train_path)
    X_train = train_frame.drop(['Image', 'label'], axis=1, inplace=False)
    all_features = np.array(X_train.columns.tolist())
    y_train = train_frame['label']
    image_train = train_frame['Image']

    test_frame = pd.read_csv(test_path)
    X_test = test_frame.drop(['Image', 'label'], axis=1, inplace=False)
    y_test = test_frame['label']
    image_test = test_frame['Image']

    test_frame_2 = pd.read_csv(test_path_2)
    X_test_2 = test_frame_2.drop(['Image', 'label'], axis=1, inplace=False)
    y_test_2 = test_frame_2['label']
    image_test_2 = test_frame_2['Image']

    model = joblib.load(model_path)

    # save predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_csv = pd.DataFrame({'image': image_train,
                                'y_truth': y_train,
                                'y_pred': y_train_pred})
    y_test_csv = pd.DataFrame({'image': image_test,
                                'y_truth': y_test,
                                'y_pred': y_test_pred})
    y_train_csv.to_csv(save_result_root + 'train_result.CSV', index=False)
    y_test_csv.to_csv(save_result_root + 'test_result.CSV', index=False)

    y_train_scores = model.decision_function(X_train)
    y_test_scores = model.decision_function(X_test)
    train_auc = roc_auc_score(y_train, y_train_scores)
    test_auc = roc_auc_score(y_test, y_test_scores)

    # get selected features
    selected_features = all_features[model.named_steps['univariate_select'].get_support()]
    selected_features = selected_features[model.named_steps['feature_select'].get_support()]
    sum = 0
    for i in model.named_steps['feature_select'].get_support():
        if i == False:
            sum += 1
    print(sum)

    # save performance
    with open(save_result_root + 'result.txt', 'a') as result_file:
        result_file.write("Train set confusion matrix is : {} \n".format
                          (confusion_matrix(y_train, y_train_pred)))
        result_file.write("Test set confusion matrix is : {} \n".format
                          (confusion_matrix(y_test, y_test_pred)))
        result_file.write("Train set precision score is {} \n".format
                          (precision_score(y_train, y_train_pred)))
        result_file.write("Test set precision score is {} \n".format
                          (precision_score(y_test, y_test_pred)))
        result_file.write("Train set recall score is {} \n".format
                          (recall_score(y_train, y_train_pred)))
        result_file.write("Test set recall score is {} \n".format
                          (recall_score(y_test, y_test_pred)))

        result_file.write("Train set roc_auc score is {} \n".format(train_auc))
        result_file.write("Test set roc_auc score is {} \n".format(test_auc))

        result_file.write("Selected features are:\n")
        for i, feature_name in enumerate(selected_features):
            result_file.write("{}: {}\n".format(i + 1, str(feature_name)))

    # plot roc curve
    fpr, tpr, thresholds = roc_curve(y_train, y_train_scores)
    plt.plot(fpr, tpr, 'r-', linewidth=2, label='Train ROC Curve(AUC=' + str(round(train_auc, 3)) + ')')
    fpr, tpr, thresholds = roc_curve(y_test, y_test_scores)
    plt.plot(fpr, tpr, 'b:', linewidth=2, label='Test_1 ROC Curve(AUC=' + str(round(test_auc, 3)) + ')')

    y_test_scores_2 = model.decision_function(X_test_2)
    test_auc_2 = roc_auc_score(y_test_2, y_test_scores_2)
    fpr, tpr, thresholds = roc_curve(y_test_2, y_test_scores_2)
    plt.plot(fpr, tpr, 'g--', linewidth=2, label='Test_2 ROC Curve(AUC=' + str(round(test_auc_2, 3)) + ')')

    plt.plot([0, 1], [0, 1], 'k-.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate(recall)')
    plt.legend(loc=4)
    plt.savefig(save_result_root + 'ROC.png')


if __name__ == '__main__':
    train_path = 'E:/hypertension/process&result/results_1.0/process_features/train/train_4.CSV'
    test_path = 'E:/hypertension/process&result/results_1.0/process_features/test/test_4.CSV'
    test_path_2 = 'E:/hypertension/3.kidneynrrd(new_data)/process_features/features_4.CSV'
    model_path = 'E:/hypertension/process&result/results_1.0/results/train_result/train_4_model.pkl'
    save_result_root = 'E:/hypertension/process&result/results_1.0/results/test_result/'

    get_results(train_path, test_path, test_path_2, model_path, save_result_root)