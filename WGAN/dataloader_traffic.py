import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

# Transforming images to with a common size

transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()]
                               )

dataset = datasets.ImageFolder("data/train/", transform=transform)

dataloader_traffic = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

images, labels = next(iter(dataloader_traffic))


# Use this if you want to

# plt.imshow(np.transpose(images[0].numpy(),(1,2,0)))
# plt.show()