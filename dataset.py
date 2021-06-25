from torchvision import datasets
from torchvision import transforms

dataroot = "data/celeba"
image_size = 64

dataset = datasets.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))


if __name__ == "__main__":
    print(dataset)