import torch
import logging
import time
import argparse


class Timer:
    def __init__(self, function_name, logger):
        self.function_name = function_name
        self.begin_time = time.time()
        self.logger = logger

    def __del__(self):
        self.logger.info(
            f"{self.function_name} costs {time.time() - self.begin_time:.3f} seconds"
        )


def check_cuda_availability():
    if not torch.cuda.is_available():
        raise SystemExit("cuda is not available!")


def get_file_and_console_logger(args):
    log_levels = {
        10: logging.DEBUG,
        20: logging.INFO,
        30: logging.WARNING,
        40: logging.ERROR,
        50: logging.FATAL,
    }

    formatter = logging.Formatter(
        fmt="[%(asctime)s.%(msecs)03d][%(filename)10s:%(lineno)4d][%(levelname)5s]|%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    current_time = time.strftime("%Y:%m:%d-%H:%M:%S", time.localtime(time.time()))
    log_path = f"log/{current_time}.log"
    handler_to_file = logging.FileHandler(log_path, mode="w")
    handler_to_file.setFormatter(formatter)
    handler_to_console = logging.StreamHandler()
    handler_to_console.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(log_levels[args.log_level])
    logger.addHandler(handler_to_file)
    logger.addHandler(handler_to_console)

    return logger


def get_argparser():
    parser = argparse.ArgumentParser(description="Hello World!")
    parser.add_argument(
        "--log_level",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--action",
        type=str,
        default="train",
        choices={"train", "test"},
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--latent_size",
        type=int,
        default=128,
    )

    args = parser.parse_args()
    return args
