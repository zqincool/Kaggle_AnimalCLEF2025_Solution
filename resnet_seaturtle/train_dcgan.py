import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader.dataloader import ImageFolderDataset, get_transforms
from net_dcgan import DCGANGenerator, DCGANDiscriminator
import argparse
from tqdm import tqdm
import os

# python train_dcgan.py --data_dir dummy --generate_unknown --generator_path /root/cnn_open_set/checkpoints/dcgan_g_epoch_20.pth --num_unknown 1000 --unknown_dir /root/autodl-tmp/images/SeaTurtleID2022/database/turtles-data/data/images/gan_unknown

# Weight initialization for DCGAN
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def generate_unknown_images(generator_path, z_dim, img_channels, feature_g, num_images, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    netG = DCGANGenerator(z_dim=z_dim, img_channels=img_channels, feature_g=feature_g).to(device)
    netG.load_state_dict(torch.load(generator_path, map_location=device))
    netG.eval()
    batch_size = 64
    total = 0
    with torch.no_grad():
        while total < num_images:
            current_batch = min(batch_size, num_images - total)
            noise = torch.randn(current_batch, z_dim, 1, 1, device=device)
            fake_images = netG(noise)
            fake_images = (fake_images + 1) / 2  # [-1,1] to [0,1]
            for i in range(current_batch):
                from torchvision.utils import save_image
                save_image(fake_images[i], os.path.join(save_dir, f"unknown_{total+i:05d}.png"))
            total += current_batch
    print(f"Generated {num_images} unknown images in {save_dir}")

def train_dcgan(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transforms(args.img_size)
    dataset = ImageFolderDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    netG = DCGANGenerator(z_dim=args.z_dim, img_channels=args.img_channels, feature_g=args.feature_g).to(device)
    netD = DCGANDiscriminator(img_channels=args.img_channels, feature_d=args.feature_d).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(16, args.z_dim, 1, 1, device=device)

    for epoch in range(args.epochs):
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
        for i, (real_images, _) in enumerate(loop):
            real_images = real_images.to(device)
            b_size = real_images.size(0)
            real_label = torch.full((b_size,), 1., dtype=torch.float, device=device)
            fake_label = torch.full((b_size,), 0., dtype=torch.float, device=device)

            # Train Discriminator
            netD.zero_grad()
            output = netD(real_images)
            lossD_real = criterion(output, real_label)
            noise = torch.randn(b_size, args.z_dim, 1, 1, device=device)
            fake_images = netG(noise)
            output = netD(fake_images.detach())
            lossD_fake = criterion(output, fake_label)
            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            output = netD(fake_images)
            lossG = criterion(output, real_label)
            lossG.backward()
            optimizerG.step()

            loop.set_postfix(lossD=lossD.item(), lossG=lossG.item())

        # Optionally save generated images or models here
        if not os.path.exists('dcgan_samples'):
            os.makedirs('dcgan_samples')
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            from torchvision.utils import save_image, make_grid
            save_image(make_grid(fake, nrow=4, normalize=True), f'dcgan_samples/fake_epoch_{epoch+1}.png')
        if epoch % 5 == 0:
            if not os.path.exists('./checkpoints'):
                os.makedirs('./checkpoints')
            torch.save(netG.state_dict(), f'./checkpoints/dcgan_g_epoch_{epoch+1}.pth')
            torch.save(netD.state_dict(), f'./checkpoints/dcgan_d_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='dataset root directory')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--img_channels', type=int, default=3)
    parser.add_argument('--feature_g', type=int, default=64)
    parser.add_argument('--feature_d', type=int, default=64)
    parser.add_argument('--generate_unknown', action='store_true', help='Generate unknown images after training')
    parser.add_argument('--generator_path', type=str, default='', help='Path to trained generator weights')
    parser.add_argument('--num_unknown', type=int, default=1000, help='Number of unknown images to generate')
    parser.add_argument('--unknown_dir', type=str, default='gan_unknown', help='Directory to save unknown images')
    args = parser.parse_args()

    if args.generate_unknown:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generate_unknown_images(
            generator_path=args.generator_path,
            z_dim=args.z_dim,
            img_channels=args.img_channels,
            feature_g=args.feature_g,
            num_images=args.num_unknown,
            save_dir=args.unknown_dir,
            device=device
        )
    else:
        train_dcgan(args) 