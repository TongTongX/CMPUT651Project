from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import requests
from io import BytesIO

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

plt.figure()
response = requests.get(url='http://meme.xyz/uploads/posts/t/l-47753-anti-vaxx-kids-when-they-see-someone-doing-the-10-year-challenge.jpg')
assert response.status_code == 200
test_image = Image.open(BytesIO(response.content))
test_array = np.array(test_image)
print(test_array.shape, test_array.dtype)
plt.imshow(test_image)
plt.show()

data_transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

transformed_image = data_transform(img=test_image)
print(transformed_image.dtype, transformed_image.shape)
# print('transformed_image.shape: {}'.format(transformed_image.shape))
# plt.figure()
# plt.imshow(transformed_image)
# plt.show()