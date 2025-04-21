'''dataset loader'''
import torch.utils.data
from data.base_dataset import BaseDataset
from data.iharmony4_dataset import Iharmony4Dataset
from data.ragiharmony4_dataset import RAGIharmony4Dataset




class CustomDataset(object):
    """User-defined dataset
    
    Example usage:
        >>> from data import CustomDataset
        >>> dataset = CustomDataset(opt, is_for_train)
        >>> dataloader = dataset.load_data()
    """
    def __init__(self, opt, is_for_train):
        self.opt = opt
        if opt.dataset_mode.lower() == 'iharmony4':
            self.dataset = Iharmony4Dataset(opt, is_for_train)
            print("dataset [%s] was created" % type(self.dataset).__name__)
        elif opt.dataset_mode.lower() == 'ragiharmony4':
            self.dataset = RAGIharmony4Dataset(opt, is_for_train)
            print("dataset [%s] was created" % type(self.dataset).__name__)


        else:
            raise ValueError(opt.dataset_mode, "not implmented.")
        if is_for_train==False:
            tempbatch = 1
        else:
            tempbatch = opt.batch_size
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=tempbatch,
            shuffle=is_for_train,
            num_workers=int(opt.num_threads),
            drop_last=False)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
