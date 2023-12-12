import torch
from torch import Tensor
from torchmetrics import Metric


class WeightedPrecision(Metric):
    def __init__(self, weights=torch.tensor([1,10,50,100,300]), dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.weights = weights

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if len(preds.shape) > 1:
            preds = torch.squeeze(preds, 1)
        # preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape
        w = self.weights[target]
        self.correct += torch.sum((preds == target) * w)
        self.total += torch.sum(w)

    def compute(self):
        return self.correct.float() / self.total


if __name__ == '__main__':
    preds = torch.tensor([0, 2, 3, 4])
    target = torch.tensor([2, 2, 4, 1])
    p = WeightedPrecision()
    prec = p(preds, target)
    print(prec)
