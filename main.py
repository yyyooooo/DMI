import argparse
import os, sys
import time
import logging, logging.config
import yaml
import importlib
from models import build_model
from solver import build_solver
import torch


def main(cfg):

    logger = logging.getLogger("main")
    logger.info(cfg)
    logger.info(f"Building Model \"{cfg['model']}\"...")
    model_cfg_file = os.path.join("config/models", cfg['model']+'.yaml')
    assert os.path.exists(model_cfg_file), f"Model config file {model_cfg_file} does not exist."
    with open(model_cfg_file, "r") as f:
        model_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = build_model(model_cfg)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join((map(str, cfg['GPUS'])))
    if len(cfg['GPUS']) > 1:
        model = torch.nn.DataParallel(model)

    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    logger.info(f"Number of parameters: {n_parameters}")

    if args.resume is not None:
        path = args.resume
        if os.path.isfile(path):

            logger.info(f"Loading checkpoint \"{path}\"...")
            checkpoint = torch.load(path, map_location='cpu')
            checkpoint = checkpoint['model']
            model_dict = model.state_dict()
            checkpoint = {k: v for k, v in checkpoint.items() if
                             k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(checkpoint)
            model.load_state_dict(model_dict)
            logger.info(f"Loaded checkpoint \"{path}\".")
        else:
            logger.warning(f"=> No checkpoint found at '{path}'.")

    logger.info(f"Creating dataloaders...")
    dataset_module = importlib.import_module("datasets."+cfg['dataset']['name'])
    loaders = dataset_module.build_loader(cfg['dataset'], args.test)

    solver = build_solver(model, loaders, cfg)

    if args.test:
        solver.test()
        sys.exit()

    solver.train()


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train a retrieval network")
    parser.add_argument(
        "--cfg", help="config file", required=True, type=str
    )
    parser.add_argument(
        "--log", help="log config file", required=True, type=str
    )
    parser.add_argument(
        "--test", action="store_true", help="run test"
    )
    parser.add_argument(
        "--resume", default=None, help="checkpoint model to resume", type=str
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    assert os.path.exists(args.cfg), f"Solver config file {args.cfg} does not exist."
    with open(args.cfg, "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    assert os.path.exists(args.log), f"Logger config file {args.log} does not exist."
    with open(args.log, "r") as f:
        log_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    log_path = os.path.join(cfg['checkpoint_dir'], cfg['name'], 'log')
    os.makedirs(log_path, exist_ok=True)
    t = 0
    save_file = os.path.join(log_path, "{}-{}_{}.log".format('eval', time.strftime("%Y_%m_%d_%H"), t))
    if os.path.exists(save_file):
        sp = save_file[:-4]
        t = int(sp.split('_')[-1])+1
        save_file = os.path.join(log_path, "{}-{}_{}.log".format('eval', time.strftime("%Y_%m_%d_%H"), t))
    log_cfg['handlers']['file']['filename'] = save_file
    logging.config.dictConfig(log_cfg)

    main(cfg)