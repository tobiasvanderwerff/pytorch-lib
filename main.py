"""
Some boilerplate code for the main function, containing common command line
arguments. Does not do anything meaningful as is.
"""

import argparse
import logging
from pathlib import Path


def main(args):
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.mkdir(exist_ok=True, parents=True)

    # set up logging
    TIMESTAMP = datetime.datetime.today().strftime("%y%m%d_%H:%M")
    log_file = checkpoint_path / f"train_{TIMESTAMP}.log"
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Writing log output to {log_file}")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--load_model", type=str,
                        help=("Load a model from a checkpoint specified by a "
                              "path with a .tar extension."))
    parser.add_argument("--data_path", type=str,
                        help="Path where the data is stored.")
    parser.add_argument("--checkpoint_path", type=str,
                        default=str(Path().cwd() / "checkpoints"),
                        help="Path to directory where to store model checkpoints.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--max_epochs_no_change", type=int, default=5,
                        help="Number of epochs until early stopping occurs.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="How many subprocesses to use for data loading.")
    parser.add_argument("--debug_mode", action="store_true", default=False,
                        help="Use a small subset of the data to enable faster debugging")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for initializing random number generators.")
    args = parser.parse_args()
    # fmt: on

    main(args)
