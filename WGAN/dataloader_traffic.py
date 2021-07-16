import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

# Transforming images to with a common size

#transform = transforms.Compose([transforms.Resize(255),
#                                transforms.CenterCrop(224),
#                                transforms.ToTensor()]
#                               )

transform =  transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

dataset = datasets.ImageFolder("data/WGAN/train/", transform=transform)

dataloader_traffic = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

images, labels = next(iter(dataloader_traffic))


# Use this if you want to

# plt.imshow(np.transpose(images[0].numpy(),(1,2,0)))
# plt.show()
