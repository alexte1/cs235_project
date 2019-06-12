# -*- coding: utf-8 -*-
from csv import reader
import math
import numpy as np
import matplotlib.pyplot as plt

def load_csv(filename):
    with open(filename, 'r') as inf:
        lines = reader(inf)
        dataset = list(lines)
        return dataset
        
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def gini_score(groups, classes):
    # weight = sum(group) / sum(sum(group))
    # Gini index = sum(sum(one_class) / sum(group))
    instances_num = sum([len(group) for group in groups])
    gini_score = 0.0
    for group in groups:
        group_num = len(group)
        if group_num == 0:
            continue
        gini_index = 1.0
        for c in classes:
            c_num = 0
            for i in group:
                if c == i[-1]:
                    c_num += 1
            c_p = c_num / group_num 
            gini_index = gini_index - c_p * c_p
        group_gini_score = gini_index * group_num / instances_num
        gini_score += group_gini_score
    return gini_score

def split_group(index, split_value, group):
    less = list()
    big = list()
    for row in group:
        if row[index] < split_value:
            less.append(row)
        else:
            big.append(row)

    return less, big

def get_split(group):
    best_index = 0
    best_split_value = 0
    best_gini = math.inf
    best_groups = None
    classes = list(set([row[-1] for row in group]))
    for i in range(len(group[0]) - 1): 
        tem_split_values = list(set([row[i] for row in group]))
        for split_value in tem_split_values:
            groups = split_group(i, split_value, group)
            gini = gini_score(groups, classes)
            if gini < best_gini:
                best_gini = gini
                best_index = i
                best_split_value = split_value
                best_groups = groups
    return {
        'index': best_index,
        'split_value': best_split_value,
        'groups': best_groups
    }   
    
# Create a terminal node value, find the majority class and use it as the label or value.
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count) 


def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'] = to_terminal(left)
        node['right'] = to_terminal(right)
        return 
    if len(left) < min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    if len(right) < min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)

def build_tree(train_group, max_depth, min_size):
    root = get_split(train_group)
    split(root, max_depth, min_size, 1)
    return root
    
def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    accuracy = correct / float(len(actual)) * 100.0
    return accuracy

def evaluate_algorithm(traindata, algorithm, testdata, *args):
    train_set = traindata
    test_set = testdata
    scores = list()
    print('lenght of test_set', len(test_set))
    predicted = algorithm(train_set, test_set, *args)
    print('length: predicted', len(predicted))
    actual = [row[-1] for row in test_set]
    pridictedaccuracy = accuracy(actual, predicted)
    scores.append(pridictedaccuracy)
    return scores

def predict(node, row):

    if isinstance(node, dict):
        index = node['index']
        split_value = node['split_value']
        if row[index] < split_value:
            return predict(node['left'], row)
        else:
            return predict(node['right'], row)
    else:
        return node

def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions

def test_DecisionTreeClassifier_depth(traindata,algorithm,testdata,max_depth,min_szie):
    depths=np.arange(1,max_depth)
    training_scores=[]
    testing_scores=list()
    train_set = traindata
    test_set = testdata
    for depth in depths:
        trainpredicted = decision_tree(train_set, train_set, depth, min_size)
        trainactual = [row[-1] for row in train_set]
        trainaccuracy = accuracy(trainactual, trainpredicted)
        training_scores.append(trainaccuracy)
        testpredicted = decision_tree(train_set, test_set, depth, min_size)
        testactual = [row[-1] for row in test_set]
        testaccuracy = accuracy(testactual, testpredicted)
        testing_scores.append(testaccuracy)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(depths,training_scores,label="traing score",marker='o')
    ax.plot(depths,testing_scores,label="testing score",marker='*')
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Classification")
    ax.legend(framealpha=0.5,loc='best')
    plt.show()

if __name__=='__main__':
    traindata = load_csv('Data/all_traindata_dt3.csv')
    testdata = load_csv('Data/all_testdata_dt3.csv')
    for i in range(len(traindata[0])):
        str_column_to_float(traindata, i)
    for i in range(len(testdata[0])):
        str_column_to_float(testdata, i)
    ##a=input('max_depth:')
    ##b=input('min_size:')
    ##max_depth = int(a)
    ##min_size = int(b)
    max_depth = 8
    min_size = 22
    scores = evaluate_algorithm(traindata, decision_tree, testdata, max_depth, min_size)
    print('Scores: %s' % scores)
    test_DecisionTreeClassifier_depth(traindata,decision_tree,testdata, 100, min_size)

