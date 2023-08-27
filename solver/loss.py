import torch
import torch.nn as nn
from utils.registry import Registry


LOSSES = Registry()

def build_loss(loss_name, *args, **kwargs):
	assert loss_name in LOSSES, f"loss name {loss_name} is not registered in registry."
	return LOSSES[loss_name](*args, **kwargs)


@LOSSES.register("info_nce")
class InfoNCELoss(nn.Module):
	def __init__(self, temperature=0.07):
		super(InfoNCELoss, self).__init__()
		self.loss_fn = nn.CrossEntropyLoss()
		self.temp = temperature

	def forward(self, ref, pos, mem):
		'''
			ref: (batch_size, d)
			pos: (batch_size, d)
			mem: (mem_size, d)
		'''
		# => (batch_size, batch_size)
		scores_ib = ref.mm(pos.t())
		# => (batch_size, mem_size)
		scores_rm = ref.mm(mem.t())
		# => (batch_size, mem_size)
		scores_pm = pos.mm(mem.t())

		# => (batch_size, batch_size+mem_size)
		scores = torch.cat((scores_ib, scores_rm), dim=1)
		scores = torch.div(scores, self.temp)
		target = torch.arange(len(ref), dtype=torch.long, device=ref.device)
		loss = self.loss_fn(scores, target)

		# => (batch_size, batch_size+mem_size)
		scores = torch.cat((scores_ib.t(), scores_pm), dim=1)
		scores = torch.div(scores, self.temp)
		target = torch.arange(len(pos), dtype=torch.long, device=pos.device)
		loss += self.loss_fn(scores, target)

		return loss
