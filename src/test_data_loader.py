from monai.data import DataLoader, Dataset, PersistentDataset, CacheDataset

class TestDataset(PersistentDataset):

    def __getitem__(self, index):
        data = self.data[index]
        print(data)
        return super().__getitem__(index)
