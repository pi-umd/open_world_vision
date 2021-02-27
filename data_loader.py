import os

from PIL import Image
from torch.utils import data


def image_loader(img_path, accimage=False):
    """Loads an image.

    Args:
        img_path (str): path to image file
        accimage (bool): whether to use accimage loader. `True` means that it would be used.
    """
    if accimage:
        import accimage
        img = accimage.Image(img_path)
    else:
        with Image.open(img_path) as img:
            img = img.convert('RGB')
    return img


class EvalDataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, data_file, accimage=False, transform=None):
        """Initialization"""
        self.transform = transform
        self.accimage = accimage
        with open(data_file) as f:
            lines = f.readlines()
            file_name, ext = os.path.splitext(data_file)
            if ext == '.txt':
                self.data_list = [line.strip('\n') for line in lines]
            else:
                self.data_list = [line.strip('\n').split(',')[0] for line in lines]

    def __len__(self):
        """Returns the total number of samples"""
        return len(self.data_list)

    def __getitem__(self, index):
        """Generates one sample of data"""
        img_path = self.data_list[index]
        try:
            img_data = image_loader(img_path, accimage=self.accimage)
            if self.transform is not None:
                img_data = self.transform(img_data)
            return index, img_data
        except FileNotFoundError:
            print(f'File not found {img_path}')
        except Exception as e:
            print(f'Error loading image: {img_path}. Error: {e}')
        return None
