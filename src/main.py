import utils
import data
import models

if __name__ == "__main__":
    utils.check_cuda_availability()

    args = utils.get_argparser()
    logger = utils.get_file_and_console_logger(args)

    dataloader = data.get_MNIST_dataloader(args)

    mnist_gan = models.GAN(args, logger)

    if args.action == "train":
        mnist_gan.train(dataloader)
    else:
        mnist_gan.test(dataloader)
