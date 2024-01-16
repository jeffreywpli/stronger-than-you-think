import logging
import numpy as np
import pprint
from sklearn import svm
from sklearn.utils import shuffle
import random
# load numpy array from csv file
from numpy import loadtxt
from numpy import savetxt
from wrench.dataset import load_dataset
from scipy.stats import mode
from sklearn import metrics
from sklearn.metrics import f1_score
import json



'''
dataset_home = '../datasets'
data = 'basketball'
extract_fn = 'bert'
model_name = 'bert-base-cased'
train_data, valid_data, test_data = load_dataset(dataset_home, data, extract_feature=True, extract_fn=extract_fn,
                                                 cache_name=extract_fn, model_name=model_name)

labels = train_data.labels + valid_data.labels + test_data.labels
train_feature = [d['feature'] for d in train_data.examples]
valid_feature = [d['feature'] for d in valid_data.examples]
test_feature = [d['feature'] for d in test_data.examples]
features = train_feature + valid_feature + test_feature
labels = np.array(labels)
features = np.array(features)

from numpy import savetxt
savetxt('../datasets/basketball/labels.csv', labels, delimiter=',')
savetxt('../datasets/basketball/features.csv', features, delimiter=',')
'''


def ws_LF_labels(features, labels, label_num):
    index_0 = 0
    index_1 = 0
    num_list = []
    label0_list = []
    feature0_list = []
    label1_list = []
    feature1_list = []
    while(index_0 < label_num):
        num = random.randint(0, 20255)
        if labels[num] == 0:
            if num not in num_list:
                num_list.append(num)
                label0_list.append(0)
                feature0_list.append(features[num])
                index_0 += 1
    while(index_1 < label_num):
        num = random.randint(0, 20255)
        if labels[num] == 1:
            if num not in num_list:
                num_list.append(num)
                label1_list.append(1)
                feature1_list.append(features[num])
                index_1 += 1
    return label0_list, label1_list, feature0_list, feature1_list



def single_pair_pred(label_num, pred_label_list, score_list,
                            label0_list, label1_list, feature0_list, feature1_list, func_count):
    for i in range(label_num):
        for j in range(label_num):        
            labels_select = []
            labels_select.append(label0_list[i])
            labels_select.append(label1_list[j])
            features_select = []
            features_select.append(feature0_list[i])
            features_select.append(feature1_list[j])
            labels_select = np.array(labels_select)
            features_select = np.array(features_select)
            features_remain = feature0_list + feature1_list
            features_remain = np.array(features_remain)
            features_remain = np.delete(features_remain, [i, (j + label_num)], axis=0)
            label_remain = label0_list + label1_list
            label_remain = np.array(label_remain)
            label_remain = np.delete(label_remain, [i, (j + label_num)])


            clf = svm.SVC(kernel="linear", C=1.0)
            clf.fit(features_select, labels_select)
            coef = clf.coef_[0]
            intercept = clf.intercept_[0]
            margin_intercept0 = np.dot(coef, feature0_list[i])
            margin_intercept1 = np.dot(coef, feature1_list[j])
            #print(margin_intercept0, margin_intercept1, intercept)
            #target_dist = abs(margin_intercept0 + intercept) / np.sqrt(np.dot(coef,coef))
            max_score = 0
            cur_bound = max_bound = 1
            while cur_bound > 0.09:
                remain_y_pred = []
                for remaining in features_remain:
                    #print("actual distance is" + str(np.dot(coef, remaining) + intercept))
                    if abs(np.dot(coef, remaining) + intercept) < (abs(margin_intercept0 + intercept)) * cur_bound:
                        remain_y_pred.append(-1)
                    else:
                        #print("actual distance for label 0 is" + str(np.dot(coef, feature) + intercept))
                        if np.linalg.norm(remaining - feature0_list[i]) < np.linalg.norm(remaining - feature1_list[j]):
                            #print("actual distance for label 0 is" + str(np.dot(coef, feature) + intercept))
                            remain_y_pred.append(0)
                        else:
                            #print("actual distance for label 1 is" + str(np.dot(coef, feature) + intercept))
                            remain_y_pred.append(1)
                cur_score = f1_score(label_remain, remain_y_pred, average='weighted')
                if cur_score > max_score:
                    max_score = cur_score
                    max_bound = cur_bound
                cur_bound -= 0.05
            score_list.append(max_score)
            y_pred = []
            for feature in features:
                #print("actual distance is" + str(np.dot(coef, feature) + intercept))
                if abs(np.dot(coef, feature) + intercept) < (abs(margin_intercept0 + intercept)) * max_bound:
                    y_pred.append(-1)
                else:
                    if np.linalg.norm(feature - feature0_list[i]) < np.linalg.norm(feature - feature1_list[j]):
                        #print("actual distance for label 0 is" + str(np.dot(coef, feature) + intercept))
                        y_pred.append(0)
                    else:
                        #print("actual distance for label 1 is" + str(np.dot(coef, feature) + intercept))
                        y_pred.append(1)
            y_pred = np.array(y_pred)
            pred_label_list = np.append(pred_label_list, np.array([y_pred]).transpose(), axis=1)
            #print(label_model.score(features, labels))

    score_list = np.array(score_list)
    sort_score = np.argsort(score_list)[-func_count:].tolist()
    pred_label_list = pred_label_list[:,sort_score]
    score_list = np.sort(score_list)[-func_count:]
    return pred_label_list, score_list


def double_pair_pred(pred_label_list, score_list,
                            label0_list, label1_list, feature0_list, feature1_list, num):
    for i in range(300):
        label0_num1 = random.randint(0, num-1)
        label0_num2 = random.randint(0, num-1)
        label1_num1 = random.randint(0, num-1)
        label1_num2 = random.randint(0, num-1)
        labels_select = np.array([label0_list[label0_num1], label0_list[label0_num2],
                            label1_list[label1_num1], label1_list[label1_num2]])
        features_select = np.array([feature0_list[label0_num1], feature0_list[label0_num2],
                            feature1_list[label1_num1], feature1_list[label1_num2]])
        
        label_model = svm.SVC(kernel="linear", C=1.0)
        label_model.fit(features_select, labels_select)
        y_pred = label_model.predict(features)
        pred_label_list = np.append(pred_label_list, np.array([y_pred]).transpose(), axis=1)
        score_list = np.append(score_list, label_model.score(features, labels))

    return pred_label_list, score_list


if __name__ == "__main__":

    labels = loadtxt('../datasets/basketball/labels.csv', delimiter=',')
    features = loadtxt('../datasets/basketball/features.csv', delimiter=',')
    label0_list, label1_list, feature0_list, feature1_list = ws_LF_labels(features, labels, 15)

    pred_label_list = np.empty((20256, 0), int)
    score_list = []
    pred_label_list, score_list  = single_pair_pred(15, pred_label_list, score_list,
                            label0_list, label1_list, feature0_list, feature1_list, 15)

    #pred_label_list, score_list  = double_pair_pred(pred_label_list, score_list,
    #                        label0_list, label1_list, feature0_list, feature1_list, 15)
    
    print(score_list, score_list.shape)
    print(pred_label_list.shape)
    
    savetxt('../datasets/basketball/pred_label_1.csv', pred_label_list, delimiter=',')
    savetxt('../datasets/basketball/score_1.csv', score_list, delimiter=',')

    pred_labels = loadtxt('../datasets/basketball/pred_label_1.csv', delimiter=',')
    print(type(pred_labels), pred_labels[0])

    
    
    basketball_valid = open("../datasets/basketball/valid.json", "r")
    valid_data = json.load(basketball_valid)
    basketball_valid.close()
    i = 17970
    for key in valid_data:
        valid_data[key]['weak_labels'] = list(map(int, pred_labels[i].tolist()))
        i = i + 1
    print(i)
    basketball_valid = open("../datasets/basketball/valid.json", "w")
    json.dump(valid_data, basketball_valid)
    basketball_valid.close()


    basketball_test = open("../datasets/basketball/test.json", "r")
    test_data = json.load(basketball_test)
    basketball_test.close()
    i = 19034
    for key in test_data:
        test_data[key]['weak_labels'] = list(map(int, pred_labels[i].tolist()))
        i = i + 1
    print(i)
    basketball_test = open("../datasets/basketball/test.json", "w")
    json.dump(test_data, basketball_test)
    basketball_test.close()


    basketball_train = open("../datasets/basketball/train.json", "r")
    train_data = json.load(basketball_train)
    basketball_train.close()
    i = 0
    for key in train_data:
        train_data[key]['weak_labels'] = list(map(int, pred_labels[i].tolist()))
        i = i + 1
    print(i)
    basketball_train = open("../datasets/basketball/train.json", "w")
    json.dump(train_data, basketball_train)
    basketball_train.close()
    


    pred_labels = pred_labels.transpose()
    pred = mode(pred_labels)[0].reshape(20256)
    print(features[0])

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(labels, pred))