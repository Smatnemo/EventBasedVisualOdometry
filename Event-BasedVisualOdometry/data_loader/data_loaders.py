from torch.utils.data import DataLoader
from torchvision import transforms as transforms
# local modules
from .dataset import MyCustomDataset
from utils.data import concat_dataset


class NpyDataLoader(DataLoader):
    """
    """
    def __init__(self, data_file, batch_size, shuffle=True, num_workers=1,
                 pin_memory=True, sequence_kwargs = {}):
        self.sequence_length = 10
        event_path, img_path = concat_dataset(data_file)
        dataset = MyCustomDataset(event_path, img_path, self.sequence_length, transforms.ToTensor())
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)