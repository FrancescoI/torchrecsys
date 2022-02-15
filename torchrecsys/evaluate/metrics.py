import torch

class Metrics:

    def hit_rate(self, y_hat, y_pred):
        """
        Given a prediction and a target, it returns the hit rate.
        """
        hit_rate = 1 if torch.sum(torch.isin(y_pred, y_hat)) >= 1 else 0

        return hit_rate

    
    def auc_score(self, positive, negative):

        """
        Given a positive and a negative example, it returns the AUC score.
        """

        auc = positive > negative

        return auc.sum() / len(positive)

