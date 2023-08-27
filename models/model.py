import random
import torch
import torch.nn as nn
from utils.registry import Registry
from torch.nn.functional import kl_div

MODELS = Registry()

def build_model(cfg):
    assert (
        cfg['name'] in MODELS
    ), f"model {cfg['name']} is not defined"
    return MODELS[cfg['name']](cfg)

@MODELS.register("DMI")
class DMI(nn.Module):
    def __init__(self, cfg):
        super(DMI, self).__init__()
        self.input_dim = cfg['input_dim']
        self.one_dim = cfg['one_dim']
        self.num_layer = cfg['num_layer']
        self.layers = self._make_layers(cfg)
        self.pred_layer = self._make_pred_layer(
            cfg)
        self.is_early_fusion = (cfg['fusion'] == 'early_fusion')
        self.is_late_fusion = (cfg['fusion'] == 'late_fusion')
        self.softmax = torch.nn.Softmax(dim=1)
        self.temperature = cfg['temperature']
        self.downsample = nn.Sequential(nn.Linear(cfg['input_dim'], cfg['output_dim']))

    def forward(self, x, lens=None, norm=True):
        if lens is not None and self.is_early_fusion:
            x = self.early_fusion(x, lens)
        out = []
        down_embedding = self.downsample(x)
        for i in range(self.num_layer):
            out.append(self.layers[i](x))
        input_embedding = torch.cat(out, dim=-1)
        temperature = self.temperature
        disentangle_loss = 0
        max_mi_loss = 0
        cnt_min = 0
        pred = self.pred_layer(input_embedding)
        final_out = pred + down_embedding
        if norm:
            final_out = nn.functional.normalize(final_out, p=2, dim=1)
            down_embedding = nn.functional.normalize(down_embedding, p=2, dim=1)
            input_embedding = nn.functional.normalize(input_embedding, p=2, dim=1)
            for i in range(self.num_layer):
                out[i] = nn.functional.normalize(out[i], p=2, dim=1)
        num_rec = [0] * self.num_layer
        for sub_feat in out:
            max_mi_loss += kl_div(self.softmax(sub_feat.detach() / temperature),
                                  self.softmax(down_embedding / temperature))
        max_mi_loss -= kl_div(self.softmax(pred.detach() / temperature),
                              self.softmax(down_embedding / temperature))
        for j in range(1, self.num_layer):
            for i in range(self.num_layer):
                if num_rec[i] < self.num_layer - 1 and num_rec[(i + j) % self.num_layer] < self.num_layer - 1:
                    disentangle_loss += kl_div(self.softmax(out[i] / temperature).log(),
                                          self.softmax(out[(i + j) % self.num_layer] / temperature), reduction='mean')
                    num_rec[i] += 1
                    num_rec[(i + j) % self.num_layer] += 1
                    cnt_min += 1
                if sum(num_rec) == (self.num_layer - 1) * self.num_layer:
                    break
            if sum(num_rec) == (self.num_layer - 1) * self.num_layer:
                break

        if self.training:
            return out, down_embedding, input_embedding, max_mi_loss, disentangle_loss, final_out
        return final_out

    def early_fusion(self, x, lens):
        _x = []
        for i in range(x.size(0)):
            if self.training and random.random() <= 0.5 and lens[i] > 10:
                if lens[i] < 15:
                    sample_num = 5
                elif lens[i] < 30:
                    sample_num = 10
                else:
                    sample_num = 15
                sample = x[i][random.sample(range(lens[i]), sample_num)]
            else:
                sample = x[i][:lens[i]]
            _x.append(torch.mean(sample, dim=0))
        _x = torch.stack(_x, dim=0)
        return _x

    def _make_layers(self, cfg):
        layers = []
        for i in range(self.num_layer):
            layers.append(
                nn.Sequential(nn.Linear(cfg['input_dim'], self.one_dim)))
        return nn.ModuleList(layers)

    def _make_pred_layer(self, cfg):
         return nn.Sequential(nn.Linear(self.one_dim*self.num_layer, self.one_dim))