class Focalloss(torch.nn.Module):
    def __init__(self, gamma=1, alpha=0.60, reduction="mean"):
        super(Focalloss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        # loss = -self.alpha * (1 - predict)**self.gamma * target * torch.log(predict) - (1- self.alpha) * predict**self.gamma*(1-target)*torch.log(1 - predict)
        zeros = torch.zeros_like(predict)
        pos_p_sub = torch.where(target > zeros, target - predict, zeros)
        neg_p_sub = torch.where(target > zeros, zeros, predict)
        per_entry_cross_ent = -self.alpha * (pos_p_sub ** self.gamma) * torch.log(
            torch.clamp(predict, 1e-8, 1.0)) - (1 - self.alpha) * (neg_p_sub ** self.gamma) * torch.log(
            torch.clamp(1.0 - predict, 1e-8, 1.0))
        return per_entry_cross_ent.sum()



