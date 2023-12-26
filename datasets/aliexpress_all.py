import os.path

import numpy as np
import datatable as dt
import torch

class AllDataset_DT(torch.utils.data.Dataset):
    """
    AliExpress Dataset
    This is a dataset gathered from real-world traffic logs of the search system in AliExpress
    Reference:
        https://tianchi.aliyun.com/dataset/dataDetail?dataId=74690
        Li, Pengcheng, et al. Improving multi-scenario learning to rank in e-commerce by exploiting task relationships in the label space. CIKM 2020.
    """
    def __init__(self, dataset_paths, jay_paths, divided_paths):
        if os.path.getsize(jay_paths[3]) == 0:
            for id in range(4):
                data = dt.fread(dataset_paths[id])
                del data['search_id']
                print(data.shape)
                ids = dt.Frame({'scenaio_id': [id for i in range(data.shape[0])]})
                data = dt.cbind([ids, data])
                data.to_jay(jay_paths[id])
                del ids

        if os.path.getsize(divided_paths[2]) == 0:
            data = dt.iread(jay_paths, tempdir='./datasets')
            data = dt.rbind(data)
            data[:, :17].to_jay(divided_paths[0])
            data[:, 17: -2].to_jay(divided_paths[1])
            data[:, -2:].to_jay(divided_paths[2])
            del data

        print('Reading divided data')
        self.categorical_data = dt.fread(divided_paths[0]).to_numpy().astype(np.int)
        self.numerical_data = dt.fread(divided_paths[1]).to_numpy().astype(np.float32)
        self.labels = dt.fread(divided_paths[2]).to_numpy().astype(np.float32)

        self.numerical_num = self.numerical_data.shape[1]
        self.field_dims = np.max(self.categorical_data, axis=0) + 1
        print(self.field_dims)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.categorical_data[index], self.numerical_data[index], self.labels[index]

