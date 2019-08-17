import torch
import torch.nn.functional as F


_score_funcs = {
    'inner': lambda h, t: (h*t).sum(-1),
    'l1': lambda h, t: -torch.abs(h - t).sum(-1),
    'l2': lambda h, t: -torch.pow(h - t, 2).sum(-1),
}


def weight_loss(loss_func):
    def weight_loss_func(*args, weights=None, **kwargs):
        loss = loss_func(*args, **kwargs)
        if weights is not None:
            while len(weights.size()) != len(loss.size()):
                weights = weights.unsqueeze(-1)
            loss *= weights
        return loss.mean()
    return weight_loss_func


@weight_loss
def bpr_loss(p_score, n_score):
    diff = p_score - n_score
    loss = -F.logsigmoid(diff)
    return loss


class Score(torch.nn.Module):

    def __init__(self, score='inner'):
        super().__init__()
        self.score_func = _score_funcs[score]

    def forward(self, r, p1_feat, p2_feat, n1_feat, n2_feat):
        p1_feat = p1_feat.unsqueeze(1)
        p2_feat = p2_feat.unsqueeze(1)

        p_score = self.score_func(p1_feat, p2_feat)
        n1_score = self.score_func(n1_feat, p2_feat)
        n2_score = self.score_func(p1_feat, n2_feat)

        loss = bpr_loss(p_score, n1_score)
        loss += bpr_loss(p_score, n2_score)

        return loss

    def predict(self, r, p1_feat, p2_feat):
        p1_feat = p1_feat.unsqueeze(1) if len(p1_feat.size()) == 2 else p1_feat
        p2_feat = p2_feat.unsqueeze(1) if len(p2_feat.size()) == 2 else p2_feat
        score = self.score_func(p1_feat, p2_feat)
        return score

