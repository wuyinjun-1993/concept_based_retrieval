import torch
from torch.utils.data import Dataset


def collate_fn(batch):
    batch_data = torch.cat(batch)
    batch_boolean = torch.zeros(batch_data.size(0), len(batch))
    
    start_idx = 0
    for idx in range(len(batch)):
        end_idx = start_idx + batch[idx].size(0)
        batch_boolean[start_idx:end_idx, idx] = 1
    return batch_data, batch_boolean

class Partitioned_vector_dataset(Dataset):
    def __init__(self, all_sub_corpus_embedding_ls):
        self.data = all_sub_corpus_embedding_ls

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]