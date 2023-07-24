
# import numpy as np
# import torchvision.transforms as transforms

# images = torch.randn(8, 4, 80, 80)

# std, mean = torch.std_mean(images, dim = (2,3))
# transform_norm = transforms.Compose([transforms.Normalize(mean, std, inplace=True)])

# normalized_images = torch.stack([transforms.Normalize(mean=mean[i], std=std[i])(img) for i, img in enumerate(images)])
# std, mean = torch.std_mean(normalized_images, dim = (2,3))

# print(normalized_images.shape)

# arr = np.random.rand(3)
# print(arr)

# mean, std = np.mean(arr), np.std(arr)
# arr = (arr - mean)/std

# print(arr)



# import os

# path = './data'

# if(os.path.exists(path)):
#     dir_contents = os.listdir(path)

#     for file_name in dir_contents:
#         print(file_name)

# else : print("no such path")

# import os

# path = './datasets'
# if(os.path.exists(path)):
#     print("yes")


# from torchinfo import summary

# from Network_Real_Burst import Burstormer
# model = Burstormer()

# summary(model, input_size=(8, 4, 80, 80))

# import sys

import torch

x = torch.randn(2, 3, 30, 30)
y = torch.randn(2, 3, 30, 30)

z = torch.cat([x, y], dim=1)
print(z.shape)

# variance = x.var(-1, keepdim = True)

# print(variance.shape)