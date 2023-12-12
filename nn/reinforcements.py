import torch
import tianshou as ts

class ReinforcementGuider(torch.Module):

    def __init__(self, in_feats) -> None:
        super().__init__()

    def forward(self, obs, state=None, info={}):
        suggest = torch.zeros(obs.shape)

        return suggest, None