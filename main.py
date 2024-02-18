import argparse
from trainer import *
from load_data import *
import os

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


def main(args):
    os.makedirs(args.dataset, exist_ok=True)
    dataset = Data(
        data_dir=args.path_dataset_folder + args.dataset,
        train_plus_valid=args.train_plus_valid,
        reverse=False,
    )
    trainer = Trainer(
        dataset=dataset,
        model=args.model_name,
        parameters=vars(args),
    )
    try:
        trainer.train_and_eval()
    except:
        print(re)
        raise RuntimeError()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ComplexEx")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--train_plus_valid", default=False)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--input_dropout", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=12.0, help="Distance parameter")
    parser.add_argument("--hidden_dropout", type=float, default=0.1)
    parser.add_argument("--feature_map_dropout", type=float, default=0.1)
    parser.add_argument("--num_of_output_channels", type=int, default=1)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--path_dataset_folder", type=str, default="data/KGs/")
    parser.add_argument("--dataset", type=str, default="WN18RR")
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of cpus used during batching"
    )
    parsed_args = parser.parse_args()
    main(args=parsed_args)
