import argparse
from trainer import *
from load_data import *
import os

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--label_smoothing", type=float, default=0.3)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--path_dataset_folder", type=str, default="data/")
    parser.add_argument("--dataset", type=str, default="WN18RR")
    parser.add_argument("--dataset", type=str, default="FB15k-237")
    parser.add_argument("--mode", type=str, default="eval")

    args = parser.parse_args()
    os.makedirs(args.dataset, exist_ok=True)

    dataset = Data(
        path=args.path_dataset_folder + args.dataset,
    )
    trainer = Trainer(
        dataset=dataset,
        parameters=vars(args),
    )

    trainer.main()
