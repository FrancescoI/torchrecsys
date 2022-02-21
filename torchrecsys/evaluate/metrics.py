import torch
import numpy as np

class Metrics:

    def hit_rate(self, y_hat, y_pred):
        """
        Given a prediction and a target, it returns the hit rate.
        """

        y_hat = y_hat.numpy()
        y_pred = y_pred.numpy()

        isin_vals = np.equal(y_hat[:, None], y_pred[:, :, None]) ### using broadcasting
        
        hit_rate = np.any(isin_vals, axis=2).sum(axis=1) 

        hit_rate = np.where(hit_rate >= 1, 1, 0)
            
        return hit_rate.sum() / y_pred.shape[0]

    
    def auc_score(self, positive, negative):

        """
        Given a positive and a negative example, it returns the AUC score.
        """

        auc = positive > negative

        return auc.sum() / len(positive)

