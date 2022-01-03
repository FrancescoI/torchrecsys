# -*- coding: utf-8 -*-

import numpy as np
from torchrecsys.helper.negative_sampling import get_negative_batch
import pandas as pd
import scipy.sparse as sp

def auc_score(positive, negative):
    
    total_auc = []
    
    positive = positive.cpu().detach().numpy()
    negative = negative.cpu().detach().numpy()

    batch_auc = (positive > negative).sum() / len(positive)
    total_auc.append(batch_auc)
        
    return np.mean(total_auc)


def evaluate(model):
    
    ### TRAIN    
    negative_train = get_negative_batch(batch=model.train, n_items=model.dataset.dataset['item_id'].max(), item_to_metadata_dict=None)
    negative_train.columns = ['user_id', 'item_id_neg', 'metadata_neg']
    
    merged_train = pd.concat([model.train, negative_train], axis=1)
    
    train_auc = []
    
    for row in merged_train.itertuples():
        
        score = np.sum(model.pred[row.user_id, row.item_id] > model.pred[row.user_id, row.item_id_neg])
        train_auc.append(score)
    
    ### TEST
    negative_test = get_negative_batch(batch=model.test, n_items=model.dataset.dataset['item_id'].max(), item_to_metadata_dict=None)
    negative_test.columns = ['user_id', 'item_id_neg', 'metadata_neg']
    merged_test = pd.concat([model.test, negative_test], axis=1)
    
    test_auc = []
    
    for row in merged_test.itertuples():
        
        score = np.sum(model.pred[row.user_id, row.item_id] > model.pred[row.user_id, row.item_id_neg])
        test_auc.append(score)
    
    print(f'Train AUC: {np.sum(train_auc) / merged_train.shape[0]}')
    print(f'Test AUC: {np.sum(test_auc) / merged_test.shape[0]}')
    
    
    
def precision_recall_k(model, k):
    
    ### TRAIN
    values = np.ones(model.train.shape[0])
    users = model.train.loc[:, 'user_id']
    items = model.train.loc[:, 'item_id']
    
    sparse_matrix = sp.csr_matrix((values, (users, items)))
    
    matrix = sparse_matrix.toarray()
    total_precision = []
    total_recall = []
    
    for index, row in enumerate(matrix):
        
        truth = np.nonzero(row)[0]
        if len(truth) > 0:
            prediction = np.argsort(-model.pred[index, :])[:k]

            n_matching = len(set(truth) & set(prediction))
            precision = n_matching / k
            recall = n_matching / len(truth)

            total_precision.append(precision)
            total_recall.append(recall)
        
    print(f'Train Precision@{k}: {np.mean(total_precision)} \nTrain Recall@{k}: {np.mean(total_recall)} \nTrain Shape: {sparse_matrix.getnnz()}')
        
    
    ### TEST
    values = np.ones(model.test.shape[0])
    users = model.test.loc[:, 'user_id']
    items = model.test.loc[:, 'item_id']
    
    sparse_matrix = sp.csr_matrix((values, (users, items)))
    
    matrix = sparse_matrix.toarray()
    total_precision = []
    total_recall = []
    
    for index, row in enumerate(matrix):
        
        truth = np.nonzero(row)[0]
        if len(truth) > 0:
            prediction = np.argsort(-model.pred[index, :])[:k]

            n_matching = len(set(truth) & set(prediction))
            precision = n_matching / k
            recall = n_matching / len(truth)

            total_precision.append(precision)
            total_recall.append(recall)
        
    print(f'\nTest Precision@{k}: {np.mean(total_precision)} \nTest Recall@{k}: {np.mean(total_recall)} \nTest Shape: {sparse_matrix.getnnz()}')