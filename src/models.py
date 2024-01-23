import torch
import imageio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torch.autograd.variable import Variable


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()

        self.latent_size = args.latent_size
        self.output_size = 28 * 28

        self.layer1 = nn.Sequential(nn.Linear(self.latent_size, 256), nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(nn.Linear(256, 512), nn.LeakyReLU(0.2))
        self.layer3 = nn.Sequential(nn.Linear(512, 1024), nn.LeakyReLU(0.2))
        self.layer4 = nn.Sequential(nn.Linear(1024, self.output_size), nn.Tanh())

    def forward(self, input: torch.tensor) -> torch.tensor:
        x = input
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(-1, 1, 28, 28)

        return x


class Discriminator(nn.Module):
    def __init__(
        self,
    ):
        super(Discriminator, self).__init__()

        self.input_size = 28 * 28

        self.layer1 = nn.Sequential(nn.Linear(self.input_size, 1024), nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(nn.Linear(1024, 512), nn.LeakyReLU(0.2))
        self.layer3 = nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU(0.2))
        self.layer4 = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

        self.criterion = nn.BCELoss()

    def forward(self, input: torch.tensor) -> torch.tensor:
        x = input.view(-1, 28 * 28)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class GAN(nn.Module):
    def __init__(self, args, logger):
        super(GAN, self).__init__()

        self.args = args
        self.logger = logger
        self.G_net = Generator(args).cuda()
        self.D_net = Discriminator().cuda()

    def _save_status_pic(
        self, real_imgs: torch.tensor, fake_imgs: torch.tensor, is_train: bool = True
    ) -> None:
        real_grid = make_grid(real_imgs[: 8 * 8]).cpu()
        split_gird = make_grid(torch.ones(8, 1, 28, 28))
        fake_grid = make_grid(fake_imgs).cpu()
        output = torch.cat((real_grid, split_gird, fake_grid), 1)
        save_image(output, f'output/{"train" if is_train else "test"}_status.png')

    def _save_progress_pic(self, input: list[torch.tensor]) -> None:
        output = [make_grid(i.cpu()) for i in input]
        output = [np.array(transforms.ToPILImage()(i)) for i in output]

        imageio.mimsave(f"output/train_progress.gif", output)

    def _save_loss_pic(self, G_losses: list[int], D_losses: list[int]) -> None:
        from matplotlib import pyplot as plt

        plt.plot(G_losses, label="G losses")
        plt.plot(D_losses, label="D losses")
        plt.legend()
        plt.savefig(f"output/train_loss.png")
        plt.close()

    def _get_ones(self, size: int) -> torch.tensor:
        return torch.ones(size, 1).cuda()

    def _get_zeros(self, size: int) -> torch.tensor:
        return torch.zeros(size, 1).cuda()

    def train(self, dataloader: dict):
        G_optimizer = torch.optim.Adam(self.G_net.parameters(), lr=2e-4)
        D_optimizer = torch.optim.Adam(self.D_net.parameters(), lr=2e-4)
        criterion = nn.BCELoss()

        latent_size = self.args.latent_size
        G_losses = []
        D_losses = []
        process = []

        validate_input = Variable(torch.randn(8 * 8, latent_size).cuda())
        for epoch in range(self.args.epoch):
            G_loss_sum = D_loss_sum = 0
            for i, data in enumerate(dataloader["train"]):
                imgs, _ = data
                imgs = imgs.cuda()

                # train D
                output = self.D_net(imgs)
                D_real_loss = criterion(output, self._get_ones(imgs.shape[0]))
                D_optimizer.zero_grad()
                D_real_loss.backward()
                D_optimizer.step()
                noise = torch.randn(imgs.shape[0], latent_size).cuda()
                output = self.D_net(self.G_net(noise))
                D_fake_loss = criterion(output, self._get_zeros(imgs.shape[0]))
                D_optimizer.zero_grad()
                D_fake_loss.backward()
                D_optimizer.step()
                D_loss_sum += D_real_loss + D_fake_loss

                # train G
                noise = torch.randn(imgs.shape[0], latent_size).cuda()
                output = self.G_net(noise)
                G_loss = criterion(self.D_net(output), self._get_ones(imgs.shape[0]))
                G_optimizer.zero_grad()
                G_loss.backward()
                G_optimizer.step()
                G_loss_sum += G_loss

            G_losses.append(G_loss.item() / i)
            D_losses.append((D_real_loss.item() + D_fake_loss.item()) / i)
            self.logger.info(
                f"[{epoch:3d}]G_loss: {G_loss/i:.5f}, D_loss: {(D_real_loss + D_fake_loss)/i:.5f}"
            )

            validate_output = self.G_net(validate_input)
            self._save_status_pic(imgs, validate_output)
            self._save_loss_pic(G_losses, D_losses)

            process.append(validate_output)

            torch.save(self.G_net.state_dict(), f"checkpoints/Generator.pth")
            torch.save(self.D_net.state_dict(), f"checkpoints/Discriminator.pth")

        self._save_progress_pic(process)

    def test(self, dataloader: dict):
        self.G_net.load_state_dict(torch.load(f"checkpoints/Generator.pth"))
        self.D_net.load_state_dict(torch.load(f"checkpoints/Discriminator.pth"))

        latent_size = self.args.latent_size
        validate_input = Variable(torch.randn(8 * 8, latent_size).cuda())
        imgs, _ = next(iter(dataloader["train"]))

        validate_output = self.G_net(validate_input)
        self._save_status_pic(imgs, validate_output, is_train=False)
