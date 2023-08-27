import os
import shutil
import torch
import logging

from .loss import build_loss
import solver.train as Trainer
from .evaluation import svd_evaluation, svd_feature_extraction
from datasets.svd.core import View
from utils.registry import Registry

SOLVERS = Registry()
logger = logging.getLogger("solver")


def build_solver(model, loaders, cfg):
    solver_name = cfg['solver']
    assert solver_name in SOLVERS, f"solver {solver_name} is not defined"

    return SOLVERS[solver_name](model, loaders, cfg)


class BaseSolver(object):
    def __init__(self, model, loaders, cfg):
        self.model = model
        self.loaders = loaders
        self.cfg = cfg

        self.model.cuda()

    def save_checkpoint(self, state, is_best, path, filename='checkpoint.pt'):
        assert path is not None, f"Checkpoint save path should not be None type."
        os.makedirs(path, exist_ok=True)
        torch.save(state, os.path.join(path, filename))
        if is_best:
            shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'model_best.pt'))


@SOLVERS.register("svd.baseline")
class SVDBaselineSolver(BaseSolver):
    def __init__(self, model, loaders, cfg):
        BaseSolver.__init__(self, model, loaders, cfg)
        self.best_100_score = 0
        self.best_100_epoch = None
        self.best_inf_score = 0
        self.best_inf_epoch = None
        self.cur_epoch = None


    def extractor(self, new_view_config):
        _extractor = getattr(self, self.cfg['dataset']['name']+'_extractor', None)
        if _extractor is None:
            logger.error(f"Don't know how to extract features for {self.cfg['dataset']['name']} dataset, PASS!")
        _extractor(new_view_config)

    def svd_extractor(self, new_view_config):
        new_view = View(new_view_config, mode='w')
        new_view.create(self.model.output_dim, self.loaders['test_loader'].query_ids, self.loaders['test_loader'].labeled_ids, self.loaders['test_loader'].unlabeled_ids)
        svd_feature_extraction(
            self.model, 
            self.loaders['test_loader'].query_loader,
            self.loaders['test_loader'].labeled_loader,
            self.loaders['test_loader'].unlabeled_loader,
            new_view
        )
        new_view.dump()


    def test(self):
        evalr = svd_evaluation(
            self.model, 
            self.loaders['test_loader'].query_loader,
            self.loaders['test_loader'].labeled_loader,
            self.loaders['test_loader'].unlabeled_loader,
            self.loaders['test_loader'].test_groundtruth,
            self.cfg['metric_topk'],
            os.path.join(self.cfg['checkpoint_dir'], self.cfg['name']),
        )
        return evalr

    def train(self):
        optimizer_cls = getattr(torch.optim, self.cfg['optimizer']['name'], None)
        one_epoch_train = getattr(Trainer, self.cfg['trainer'],None)
        if optimizer_cls is None:
            logger.error(f"Unknown optimizer {self.cfg['optimizer']['name']}.")
            return
        self.optimizer = optimizer_cls(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            *self.cfg['optimizer']['args'],
            **self.cfg['optimizer']['kwargs']
        )

        lr_scheduler_cls = getattr(torch.optim.lr_scheduler, self.cfg['lr_scheduler']['name'], None)
        if lr_scheduler_cls is None:
            logger.error(f"Unknown lr scheduler {self.cfg['lr_scheduler']['name']}.")
            return
        self.lr_scheduler = lr_scheduler_cls(
            self.optimizer,
            *self.cfg['lr_scheduler']['args'],
            **self.cfg['lr_scheduler']['kwargs'],
        )

        self.loss_fn = build_loss(
            self.cfg['loss']['name'],
            *self.cfg['loss']['args'],
            **self.cfg['loss']['kwargs']
        )

        for epoch in range(1, self.cfg['epochs']+1):
            self.cur_epoch = epoch
            one_epoch_train(
                self.model,
                self.loaders['baseline_loader'].pair_loader,
                self.loaders['baseline_loader'].negative_loader,
                self.optimizer,
                self.loss_fn,
                epoch,
                self.cfg['epochs'],
                self.cfg
            )
            self.lr_scheduler.step()

            self.save_checkpoint(
                {
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                },
                False,
                os.path.join(self.cfg['checkpoint_dir'], self.cfg['name']),
                filename=f"checkpoint_{epoch}.pt"
            )

