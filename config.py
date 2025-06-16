import argparse
import os


def get_config():
    parser = argparse.ArgumentParser(description=os.path.basename(os.path.dirname(__file__)))

    parser.add_argument("--dataset", type=str, default="nuswide", help="nuswide/flickr/coco/cifar")
    parser.add_argument("--backbone", type=str, default="resnet50", help="see network.py")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs to train for")
    parser.add_argument("--n_workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--batch_size", type=int, default=128, help="input batch size")
    parser.add_argument("--optimizer", type=str, default="adam", help="adam/rmsprop/adamw/sgd")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=4e-4, help="weight decay")
    parser.add_argument("--data_dir", type=str, default="/home/virtue/singlemodal/Dataset",
                        help="directory to dataset")
    parser.add_argument("--save_dir", type=str, default="./output", help="directory to output results")
    parser.add_argument("--n_classes", type=int, default=10, help="number of dataset classes")
    parser.add_argument("--n_bits", type=int, default=16, help="length of hashing binary")
    parser.add_argument("--topk", type=int, default=None, help="mAP@topk")

    # scheduler
    # parser.add_argument("--tau", default=[80], nargs="+", type=int, help="milestones for scheduler")
    # parser.add_argument("--gamma", default=0.3, type=float, help="gamma for scheduler")

    parser.add_argument("--tasks", default=["cls", "aux"], nargs="+", type=str, help="Name of [main task, aux. task]")
    parser.add_argument(
        "--task_p",
        nargs="+",
        type=float,
        default=[1, 0.8],
        help="Prob. of [main task, aux. task] to be included in one iteration.",
    )
    parser.add_argument(
        "--aux_dim",
        default=32,
        type=int,
        help="Output embedding sizes of the respective embeddings. List of values for [main task<-n_bits, aux. task].",
    )

    ### Adversarial Loss function parameters (Projection Network R)
    parser.add_argument("--lr_mutual", default=1e-5, type=float, help="learning rate for adversarial loss")
    parser.add_argument("--wd_mutual", default=1e-6, type=float, help="weight decay for adversarial loss")
    parser.add_argument(
        "--adversarial",
        default="cls-aux",
        type=str,
        help="Directions of adversarial loss ['target-source']: 'cls-aux' (as used in the paper) and 'aux-cls'. Can contain both directions.",
    )
    parser.add_argument(
        "--adv_weight",
        default=0.01,
        type=float,
        help="Weighting parameter for adversarial loss. Needs to be the same length as the number of adv. loss directions.",
    )
    parser.add_argument(
        "--adv_dim", default=512, type=int, help="Dimension of linear layers in adversarial projection network."
    )

    ### Interclass Mining: Parameters
    parser.add_argument(
        "--n_clusters", default=140, type=int, help="Number of clusters for auxiliary inter-class mining task."
    )
    parser.add_argument(
        "--cluster_update_freq",
        default=6,
        type=int,
        help="Number of epochs to train before updating cluster labels. E.g. 1 -> every other epoch.",
    )

    ### DistanceWeightedMiner
    parser.add_argument(
        "--miner_rho_distance_lower_cutoff",
        default=0.5,
        type=float,
        help="Lower cutoff on distances - values below are sampled with equal prob.",
    )
    parser.add_argument(
        "--miner_rho_distance_upper_cutoff",
        default=1.4,
        type=float,
        help="Upper cutoff on distances - values above are IGNORED.",
    )
    parser.add_argument(
        "--miner_rho_distance_cp", default=0.0, type=float, help="Probability to replace a negative with a positive."
    )

    # TripletMarginMiner
    parser.add_argument("--l2_normalization", type=bool, default=True, help="F.normalize(embeddings) or not?")
    parser.add_argument("--type_of_triplets", type=str, default="all", help="all/semi-hard/hard")

    # MarginLoss
    parser.add_argument(
        "--lr_margin",
        default=[5e-4, 5e-4],
        nargs="+",
        type=float,
        help="'MARGIN:   Learning rate for beta-margin values for [main task, aux. task].",
    )
    parser.add_argument("--wd_margin", default=0, type=float, help="weight decay for margin loss")
    parser.add_argument("--type_of_distance", type=str, default="euclidean", help="cosine/euclidean/squared_euclidean")
    parser.add_argument(
        "--margin",
        type=float,
        default=[0.3, 0.3],
        nargs="+",
        help="TRIPLETS: Fixed Margin value for Triplet-based loss functions for [main task, aux. task].",
    )
    parser.add_argument(
        "--beta",
        default=[1.2, 1.2],
        nargs="+",
        type=float,
        help="MARGIN:   Initial beta-margin values for [main task, aux. task].",
    )

    parser.add_argument(
        "--miner",
        # default=["TripletMarginMiner", "DistanceWeightedMiner"],
        default=["DistanceWeightedMiner", "DistanceWeightedMiner"],
        nargs="+",
        type=str,
        help="MINER:   Miner for [main task, aux. task].",
    )

    return parser.parse_args()
