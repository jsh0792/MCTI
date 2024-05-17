import os
import numpy as np
import torch
from Dataset.MTdataset import MTDataset
from utils.tools import *
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, SequentialSampler
from utils.tools import make_weights_for_balanced_classes_split
from utils.collate import collate_MT


class MTDataLoader(DataLoader):
    def __init__(self, study, csv_dir, data_dir, split_dir, name, batch_size, fold, task, shuffle=True, num_workers=1, training=True):

        self.data_dir = data_dir
        self.dataset = MTDataset(csv_path = '%s/%s.csv' % (csv_dir, study),
										   data_dir= os.path.join(data_dir, study),
										   shuffle = False, 
										   print_info = True,
										   patient_strat= False,
										   n_bins=4,
										   label_col = 'survival_months',
										   ignore=[])
        train_dataset, val_dataset, test_dataset = self.dataset.return_splits(from_id=False, csv_path='{}/split{}.csv'.format(split_dir, fold))
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.model_name = name
        self.task = task
        
        # super().__init__(self.dataset, batch_size, shuffle, num_workers)

    def get_split_loader(self, split_dataset, weighted = True, batch_size=1):
        """
            return either the validation loader or training loader 
        """
        collate = collate_MT
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {'num_workers': 4} if device.type == "cuda" else {}

        if weighted:
            weights = make_weights_for_balanced_classes_split(split_dataset, model=self.model_name, task=self.task)
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate, **kwargs)    
        else:
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate, **kwargs)

        return loader

    def get_dataloader(self):
        train_dataloader = self.get_split_loader(self.train_dataset, weighted=True)
        val_dataloader = self.get_split_loader(self.val_dataset, weighted=False)
        test_dataloader = self.get_split_loader(self.test_dataset, weighted=False)
        return train_dataloader, val_dataloader, test_dataloader

    def get_gene_num(self):
        return self.train_dataset.get_gene_num()
    
    def get_class_num(self):
        return self.train_dataset.get_class_num()
