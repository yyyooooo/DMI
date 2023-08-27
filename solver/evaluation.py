import json
import torch
import logging
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from utils.metric import APScorer, SmoothedValue


logger = logging.getLogger("solver.evaluation")


@torch.no_grad()
def svd_feature_extraction(model, query_loader, labeled_loader, unlabeled_loader, new_view):
    logger.info("Start feature extraction.")
    model.eval()
    logger.info("Processing Query Samples...")
    with tqdm(total=len(query_loader)) as bar:
        for idx, batch in enumerate(query_loader):
            data, lens, ids = batch
            
            data = data.cuda()
            lens = lens.cuda()
            
            feats = model(data, lens)
            feats = feats.cpu().numpy()

            for i, vid in enumerate(ids):
                new_view[vid] = feats[i]

            bar.update(1)

    logger.info("Processing Labeled Samples...")
    with tqdm(total=len(labeled_loader)) as bar:
        for idx, batch in enumerate(labeled_loader):
            data, lens, ids = batch
            
            data = data.cuda()
            lens = lens.cuda()
            
            feats = model(data, lens)
            feats = feats.cpu().numpy()

            for i, vid in enumerate(ids):
                new_view[vid] = feats[i]

            bar.update(1)

    logger.info("Processing Unlabeled Samples...")
    with tqdm(total=len(unlabeled_loader)) as bar:
        for idx, batch in enumerate(unlabeled_loader):
            data, lens, ids = batch
            
            data = data.cuda()
            lens = lens.cuda()
            
            feats = model(data, lens)
            feats = feats.cpu().numpy()

            for i, vid in enumerate(ids):
                new_view[vid] = feats[i]

            bar.update(1)


@torch.no_grad()
def svd_evaluation(model, query_loader, labeled_loader, unlabeled_loader, test_groundtruth, topk, record_file):
    logger.info(f"Initializing SVD Evaluator...")
    evaluator = SVDEvaluator(test_groundtruth, topk)

    logger.info("Start evaluation.")
    model.eval()

    logger.info("Processing Query Samples...")
    qfeats = []
    qids = []
    with tqdm(total=len(query_loader)) as bar:
        for idx, batch in enumerate(query_loader):
            data, lens, ids = batch
            data = data.cuda()
            if lens is not None:
                lens = lens.cuda()
            
            feats = model(data, lens)

            qfeats.append(feats)
            qids += ids

            bar.update(1)

    qfeats = torch.cat(qfeats, dim=0)
    evaluator.handle_query(qfeats, qids)

    logger.info("Processing Labaled Samples.")
    with tqdm(total=len(labeled_loader)) as bar:
        for idx, batch in enumerate(labeled_loader):
            data, lens, ids = batch

            data = data.cuda()
            if lens is not None:
                lens = lens.cuda()

            feats = model(data, lens)

            evaluator.handle_labeled(feats, ids)

            bar.update(1)

    logger.info("Processing Features of UnLabaled Samples.")
    with tqdm(total=len(unlabeled_loader)) as bar:
        for idx, batch in enumerate(unlabeled_loader):
            data, lens, ids = batch

            data = data.cuda()
            if lens is not None:
                lens = lens.cuda()

            feats = model(data, lens)

            evaluator.handle_unlabeled(feats, ids)

            bar.update(1)

    # eval result
    evalr = evaluator.score()
    logger.info(" | ".join([f"{k} mAP: {v:.4f}" for k, v in evalr.items()]))

    # dump evaluation result to file
    evaluator.dump(record_file+'.json')

    return evalr


class SVDEvaluator(object):
    def __init__(self, test_groundtruth, topk):
        self.gdtruth = test_groundtruth
        self.topk = topk

        self.ranks = defaultdict(list)
        self.scorer = APScorer()

        self.res = OrderedDict()
        for k in self.topk:
            self.res["top-"+str(k)] = SmoothedValue(window_size=None)
        self.res["top-inf"] = SmoothedValue(window_size=None)

    def handle_query(self, feats, ids):
        self.qfeats = feats
        self.qids = ids
        logger.info(f"Added to evaluator, totally {len(self.qids)} queries.")

    def handle_labeled(self, feats, ids):
        sims = torch.mm(self.qfeats, feats.t()).cpu().tolist()
        for i, qid in enumerate(self.qids):
            sim = []
            cids = []
            for j, cid in enumerate(ids):
                if cid in self.gdtruth[qid]:
                    sim.append(sims[i][j])
                    cids.append(cid)
            self.ranks[qid] += list(zip(sim, cids, [self.gdtruth[qid][cid] for cid in cids]))   

    def handle_unlabeled(self, feats, ids):
        sims = torch.mm(self.qfeats, feats.t()).cpu().tolist()
        for i, qid in enumerate(self.qids):
            self.ranks[qid] += list(zip(sims[i], ids, [0]*len(ids)))

    def score(self):
        self.aps = defaultdict(OrderedDict)
        for qid in self.qids:
            self.ranks[qid].sort(key=lambda x: x[0], reverse=True)
            for k in self.topk:
                sorted_labels = []
                for i in self.ranks[qid][:k]:
                    sorted_labels.append(i[2])
                ap = self.scorer.score(sorted_labels)
                self.aps[qid]["top-"+str(k)] = ap
                self.res["top-"+str(k)].update(ap)

            sorted_labels = []
            for i in self.ranks[qid]:
                sorted_labels.append(i[2])
            ap = self.scorer.score(sorted_labels)
            self.aps[qid]["top-inf"] = ap
            self.res["top-inf"].update(ap)

        return {k: v.avg for k, v in self.res.items()}

    def dump(self, record_file, topk=100):
        record = list()
        for qid in self.qids:
            item = {'qid': qid, 'ap': self.aps[qid], 'ranking': [], 'positive': []}
            for score, vid, label in self.ranks[qid][:topk]:
                d = {
                    'score': score,
                    'id': vid,
                    'label': label
                }
                item['ranking'].append(d)
            for cid, ispos in self.gdtruth[qid].items():
                if ispos:
                    item['positive'].append(cid)
            record.append(item)
        record.sort(key=lambda x: x['ap']['top-inf'])
        with open(record_file, 'w') as f:
            f.write(json.dumps(record, sort_keys=True, indent=4))
