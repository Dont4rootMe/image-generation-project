from PIL import Image
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.data_utils import make_dataset
from utils.class_registry import ClassRegistry
import numpy as np


datasets_registry = ClassRegistry()


@datasets_registry.add_to_registry(name="base_dataset")
class BaseDataset(Dataset):
    def get_normalization_params(self, samples_to_normalize=None):
        mean, std = [], []
        if samples_to_normalize is not None:
            np.random.seed(42)
            indexes = np.random.choice(len(self.paths), np.minimum(samples_to_normalize, len(self.paths)))
        else:
            indexes = np.arange(len(self.paths))

        for ind in tqdm(indexes, desc='data normalization'):
            image = self[ind]['images'].squeeze()
            mean.append(np.array(image).mean((1, 2)))
            std.append(np.array(image).std((1, 2)))

        return np.mean(mean, axis=0), np.mean(std, axis=0)

    def __init__(self,
                 root,
                 transforms=None,
                 samples_to_normalize=None,
                 mean=None, std=None,
                 normalize_data=False):
        self.paths = make_dataset(root)
        self.transforms = transforms if transforms is not None else lambda x: x

        # if normalization is not needed then just let data be as is
        if not normalize_data:
            self.mean = torch.tensor([0., 0., 0.])
            self.std = torch.tensor([1., 1., 1.])

        # otherwise if mean and std are provided then use them
        elif mean is not None and std is not None:
            self.mean = torch.tensor(mean)
            self.std = torch.tensor(std)

        # otherwise calculate them from data
        else:
            # initialize normalization parameters with zeros and ones
            self.mean = torch.tensor([0., 0., 0.])
            self.std = torch.tensor([1., 1., 1.])
            # then calculate them from data
            self.mean, self.std = self.get_normalization_params(samples_to_normalize)

    def __getitem__(self, ind):
        path = self.paths[ind]
        image = Image.open(path).convert("RGB")
        image = (self.transforms(image) - self.mean[:, None, None]) / self.std[:, None, None]

        return {"images": image}

    def __len__(self):
        return len(self.paths)
