import matplotlib.pyplot as plt
import numpy as np
import torch

def denormalize(img, mean=None, std=None):
    """
    img: (C, H, W)
    """
    if mean is not None and std is not None:
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
        img = img * std + mean
    return img


def plot_batch(loader, class_map=None, mean=None, std=None, n=4):
    imgs, masks = next(iter(loader))  # one batch

    imgs = imgs[:n]
    masks = masks[:n]

    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))

    if n == 1:
        axes = [axes]

    for i in range(n):
        img = imgs[i].cpu()
        mask = masks[i].cpu()

        # denormalize if needed
        img = denormalize(img, mean, std)

        # convert to HWC
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        # plot image
        axes[i][0].imshow(img)
        axes[i][0].set_title("Image")
        axes[i][0].axis("off")

        # plot mask
        axes[i][1].imshow(mask.numpy(), cmap="tab20")
        axes[i][1].set_title("Mask")
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.show()