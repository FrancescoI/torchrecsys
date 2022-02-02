# -*- coding: utf-8 -*-

import scipy.sparse as sp
import numpy as np

class EASE():
    
    def __init__(self, dataset, split_train_test=True):
        
        self.dataset = dataset
        self.item_dict = dataset.dataset.set_index('item_id')['product'].to_dict()
        self.split_train_test = split_train_test
        self._split_train_test()
        
    
    def _split_train_test(self):
        
        if self.split_train_test:
        
            print('|== Splitting Train/Test ==|')
            
            data = self.dataset.dataset.iloc[np.random.permutation(len(self.dataset.dataset))]
            self.train = data.iloc[:int(len(data) * 0.9)]
            self.test = data.iloc[int(len(data) * 0.9):]
            
            print(f'Shape Train: {len(self.train)} \nShape Test: {len(self.test)}')
            
        else:
            
            self.train = self.dataset.dataset
            
    
    def fit(self, lambda_: float = 0.5, implicit=True):
        
        if implicit:
            values = np.ones(self.train.shape[0])
        else:
            values = self.train.loc[:, 'action']
        
        users = self.train.loc[:, 'user_id']
        items = self.train.loc[:, 'item_id']
        
        matrix = sp.csr_matrix((values, (users, items)))
        self.matrix = matrix
        
        ### Weight Bij are
        ### 0s if i=j (diagonal)
        ### -Pij / Pjj otherwise
        ### where P = Xt * X - lambda*I
        
        g = matrix.T.dot(matrix).toarray() 
        
        diagonal = np.diag_indices(g.shape[0])
        g[diagonal] += lambda_ ### => gives P
        
        p = np.linalg.inv(g)
        
        b = p / (-np.diag(p)) ### => gives Bij
        b[diagonal] = 0       ### and sets diagonal to 0
        
        self.b = b
        self.pred = matrix.dot(b)
        
    
    def predict(self, user_id, k):
        
        user_prediction = self.pred[user_id, :]
        item_ranked = np.argsort(-user_prediction)[:k]
        
        item_ranked = [self.item_dict[item] for item in item_ranked]
        
        return item_ranked
    
    
    def get_similarity(self, k=10):
        
        similarity = {}
        
        for idx, row in enumerate(self.b):
            
            sorted_index = np.argsort(-row)[:k]
            sorted_codes = [self.item_dict[item] for item in sorted_index]
            similarity.update({self.item_dict[idx]: sorted_codes})
            
        return similarity
    
    
