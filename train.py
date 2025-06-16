import json
import os
import time
from copy import deepcopy

import numpy as np
import torch
from loguru import logger
from timm.utils import AverageMeter, random_seed

from BaseLine.utils import build_optimizer
from _data import build_loader, get_topk, get_class_num
from _utils import prediction, mean_average_precision, calc_learnable_params, init
from config import get_config
from loss import MarginLoss, MutualLoss
from network import build_model, ResNet50Mod
from utils import deep_cluster, prepare_for_train, update_labels
from save_mat import Save_mat


def train_epoch(args, dataloaders, net, criteria, optimizer, epoch):
    tic = time.time()

    stat_meters = {}
    for x in ["n_cls_triplets", "n_aux_triplets", "cls-loss", "aux-loss", "adv-loss", "loss", "mAP"]:
        stat_meters[x] = AverageMeter()

    dataloader_collection = [dataloaders[task] for task in args.tasks]
    data_iterator = zip(*dataloader_collection)

    net.train()
    for i, data in enumerate(data_iterator):
        for j, task in enumerate(args.tasks):
            """
            probability of run cls: 100%
            probability of run adv: 80%
            """
            run_step = np.random.choice(2, p=[1 - args.task_p[j], args.task_p[j]])
            if run_step:
                ### Extract cls/aux Embedding
                features = net(data[j][0].cuda())
                labels = data[j][1].cuda()

                ### Basic DML Loss
                loss, n_triplets = criteria[task](features[task], labels)
                stat_meters[f"{task}-loss"].update(loss)
                stat_meters[f"n_{task}_triplets"].update(n_triplets)

                ### Mutual Information Loss between both embeddings
                target, source = args.adversarial.split("-")
                mut_info_loss = criteria["adv"](features[target], features[source])
                stat_meters["adv-loss"].update(mut_info_loss)

                loss = loss + args.adv_weight * mut_info_loss
                stat_meters["loss"].update(loss)

                ### Gradient Computation and Parameter Updating
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # to check overfitting
                if j == 0:
                    q_cnt = labels.shape[0] // 10
                    map_v = mean_average_precision(
                        features["cls"][:q_cnt].sign(), features["cls"][q_cnt:].sign(), labels[:q_cnt], labels[q_cnt:]
                    )
                    stat_meters["mAP"].update(map_v)

    toc = time.time()
    stat_str = ""
    for x in stat_meters.keys():
        stat_str += f"[{x}:{stat_meters[x].avg:.1f}]" if "n_" in x else f"[{x}:{stat_meters[x].avg:.4f}]"
    logger.info(
        f"[Training][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][time:{(toc - tic):.3f}]{stat_str}"
    )


def train_val(args, train_loader, query_loader, dbase_loader):
    # setup net
    net, out_idx = build_model(args, True)

    # train_loaders = {"cls": train_loader}
    train_loaders = prepare_for_train(args, net)
    train_loaders["cls"] = train_loader

    # setup criterion
    criteria = {
        "cls": MarginLoss(args),
        "aux": MarginLoss(args, "aux"),
        "adv": MutualLoss(args),
    }

    logger.info(
        f"number of learnable params: {calc_learnable_params(net, criteria['cls'], criteria['aux'], criteria['adv'])}"
    )

    ### Move learnable parameters to GPU
    for _, loss in criteria.items():
        loss.cuda()

    # setup optimizer
    to_optim = [
        {"params": net.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": criteria["cls"].parameters(), "lr": args.lr_margin[0], "weight_decay": args.wd_margin},
        {"params": criteria["aux"].parameters(), "lr": args.lr_margin[1], "weight_decay": args.wd_margin},
        {"params": criteria["adv"].parameters(), "lr": args.lr_mutual, "weight_decay": args.wd_mutual},
    ]
    optimizer = build_optimizer(args.optimizer, to_optim)

    # setup scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.tau, gamma=args.gamma)

    # training process
    best_map = 0.0
    best_epoch = 0
    best_checkpoint = None
    count = 0
    cluster_update_counter = 0
    for epoch in range(args.n_epochs):
        # print("running with learning rates {}".format(" | ".join("{}".format(x) for x in scheduler.get_lr())))

        train_epoch(args, train_loaders, net, criteria, optimizer, epoch)

        if cluster_update_counter == args.cluster_update_freq:
            new_shared_labels = deep_cluster(train_loaders["gen"], net, args.n_clusters)
            update_labels(train_loaders["aux"].dataset, new_shared_labels)
            cluster_update_counter = 0
        else:
            cluster_update_counter += 1

        # scheduler.step()

        # we monitor mAP@topk validation accuracy every 5 epochs
        # and use early stopping patience of 10 to get the best params of models
        if (epoch + 1) % 1 == 0 or (epoch + 1) == args.n_epochs:
            qB, qL = prediction(net, query_loader, out_idx)
            rB, rL = prediction(net, dbase_loader, out_idx)
            map_v = mean_average_precision(qB, rB, qL, rL, args.topk)
            map_k = "" if args.topk is None else f"@{args.topk}"
            # del qB, qL, rB, rL
            logger.info(
                f"[Evaluating][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][best-mAP{map_k}:{best_map:.4f}][mAP{map_k}:{map_v:.4f}][count:{0 if map_v > best_map else (count + 1)}]"
            )

            if map_v > best_map:
                best_map = map_v
                best_epoch = epoch
                best_checkpoint = deepcopy(net.state_dict())
                count = 0
                Save_mat(epoch=epoch, output_dim=args.n_bits, datasets=args.dataset,
                         query_labels=qL,
                         retrieval_labels=rL,
                         query_img=qB,
                         retrieval_img=rB,
                         save_dir='.',
                         mode_name="resnet50", map=map_v)
            # del qB, qL, rB, rL
                # print(f'Precision Recall Curve data:\n"DSH":[{P},{R}],')
                # f.write('PR | Epoch %d | ' % (epoch))
                # for PR in range(len(P)):
                #     f.write('%.5f %.5f ' % (P[PR], R[PR]))
                # f.write('\n')
            # del qB, qL, rB, rL

            else:
                count += 1
                if count == 10:
                    logger.info(
                        f"without improvement, will save & exit, best mAP: {best_map}, best epoch: {best_epoch}"
                    )
                    torch.save(best_checkpoint, f"{args.save_dir}/e{best_epoch}_{best_map:.3f}.pkl")
                    break
    if count != 10:
        logger.info(f"reach epoch limit, will save & exit, best mAP: {best_map}, best epoch: {best_epoch}")
        torch.save(best_checkpoint, f"{args.save_dir}/e{best_epoch}_{best_map:.3f}.pkl")

    return best_epoch, best_map


def prepare_loaders(args, bl_func):
    train_loader, query_loader, dbase_loader = (
        bl_func(
            args.data_dir,
            args.dataset,
            "train",
            None,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            drop_last=True,
        ),
        bl_func(args.data_dir, args.dataset, "test", None, batch_size=args.batch_size, num_workers=args.n_workers),
        bl_func(args.data_dir, args.dataset, "database", None, batch_size=args.batch_size, num_workers=args.n_workers),
    )
    return train_loader, query_loader, dbase_loader


def main():
    init("0")
    args = get_config()

    dummy_logger_id = None
    rst = []
    for dataset in [ "coco", "flickr"]:
        print(f"processing dataset: {dataset}")
        args.dataset = dataset
        args.n_classes = get_class_num(dataset)
        args.topk = get_topk(dataset)

        train_loader, query_loader, dbase_loader = prepare_loaders(args, build_loader)

        for hash_bit in [16]:
        # for hash_bit in [32]:
            random_seed()
            print(f"processing hash-bit: {hash_bit}")
            args.n_bits = hash_bit
            # args.aux_dim = hash_bit  # TODO: just4test

            args.save_dir = f"./output/{args.backbone}/{dataset}/{hash_bit}"
            os.makedirs(args.save_dir, exist_ok=True)
            if any(x.endswith(".pkl") for x in os.listdir(args.save_dir)):
                # raise Exception(f"*.pkl exists in {args.save_dir}")
                print(f"*.pkl exists in {args.save_dir}, will pass")
                continue

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f"{args.save_dir}/train.log", rotation="500 MB", level="INFO")

            with open(f"{args.save_dir}/config.json", "w+") as f:
                json.dump(vars(args), f, indent=4, sort_keys=True)

            best_epoch, best_map = train_val(args, train_loader, query_loader, dbase_loader)
            rst.append({"dataset": dataset, "hash_bit": hash_bit, "best_epoch": best_epoch, "best_map": best_map})
    for x in rst:
        print(
            f"[dataset:{x['dataset']}][bits:{x['hash_bit']}][best-epoch:{x['best_epoch']}][best-mAP:{x['best_map']:.3f}]"
        )


if __name__ == "__main__":
    main()
