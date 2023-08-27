import torch
import logging
from utils.metric import MetricLogger


logger = logging.getLogger("solver.train")

def svd_train_one_epoch(model, pair_loader, negative_loader, optimizer, loss_fn, epoch, total_epochs, cfg=None):
    model.train()
    metric_logger = MetricLogger(logger, delimiter="    ")
    header = f'Epoch: [{epoch}/{total_epochs}]'
    log_freq = len(pair_loader) // 4
    alpha = cfg["loss"]["alpha"]
    max_mi_weight = cfg["loss"]["max_factor"]
    min_mi_weight = cfg["loss"]["min_factor"]
    direct_loss_weight = cfg["loss"]["direct_factor"]
    logger.info("第%d轮的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
    for batch_idx, batch in metric_logger.log_every(zip(pair_loader, negative_loader), log_freq, header=header,
                                                    iterations=len(pair_loader)):
        rp, rp_lens, _ = batch[0]
        n, n_lens, _ = batch[1]
        rp = rp.cuda()
        rp_lens = rp_lens.cuda()
        n = n.cuda()
        n_lens = n_lens.cuda()
        rp,down_embedding,input_embedding, max_mi_loss1, decouple_loss1, rpcat= model(rp, rp_lens)
        n,n_down_embedding,n_input_embedding, max_mi_loss2, decouple_loss2, ncat = model(n, n_lens)
        max_mi_loss1 = torch.mean(max_mi_loss1)
        max_mi_loss2 = torch.mean(max_mi_loss2)
        decouple_loss1 = torch.mean(decouple_loss1)
        decouple_loss2 = torch.mean(decouple_loss2)
        aus_loss1 = max_mi_weight*max_mi_loss1-min_mi_weight*decouple_loss1
        aus_loss2 = max_mi_weight*max_mi_loss2-min_mi_weight*decouple_loss2
        au_loss = (aus_loss1+aus_loss2)*alpha
        loss = 0
        for i in range(len(rp)):
            loss += loss_fn(rp[i][:len(rp[i]) // 2], rp[i][len(rp[i]) // 2:], n[i])
        loss += loss_fn(down_embedding[:len(down_embedding) // 2], down_embedding[len(down_embedding) // 2:], n_down_embedding)
        loss += loss_fn(input_embedding[:len(input_embedding) // 2], input_embedding[len(input_embedding) // 2:],
                        n_input_embedding)
        dir_loss = loss_fn(rpcat[:len(rpcat) // 2], rpcat[len(rpcat) // 2:], ncat)
        loss += dir_loss*direct_loss_weight

        loss += au_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metric_logger.update(dir_loss=dir_loss.cpu().item(), loss=loss.cpu().item(), au_loss=au_loss.cpu().item())