import torch

class Metrics:

    def hit_rate(self, y_hat, y_pred):
        """
        Given a prediction and a target, it returns the hit rate.
        """

        hit_rates = 0

        for y_h, y_p in zip(y_hat, y_pred):
            hit_rate = 1 if torch.isin(y_p, y_h).sum() >= 1 else 0
            hit_rates += hit_rate
            
        return hit_rates / y_pred.shape[0]

    
    def auc_score(self, positive, negative):

        """
        Given a positive and a negative example, it returns the AUC score.
        """

        auc = positive > negative

        return auc.sum() / len(positive)

