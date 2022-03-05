import numpy as np


class LabelDict():
    def __init__(self, labels, unk_label = '[UNK]'):
        self.unk_label = unk_label
        if unk_label not in labels:
            self.labels = [unk_label] + labels
        else:
            self.labels = labels
            
        assert len(self.labels) == len(set(self.labels)), "ERROR: repeated labels appeared!"

    def __getitem__(self, idx):
        #print(idx,type(idx))
        if isinstance(idx, list):
            return [self.__getitem__(i) for i in idx]
        elif isinstance(idx, str):
            if idx in self.labels:
                return self.labels.index(idx)
            else:
                return self.labels.index(self.unk_label)
        elif isinstance(idx, int):
            return self.labels[idx]

        print("Warning: unknown indexing type!")
        return None

    def __len__(self):
        return len(self.labels)
    
    def save_dict(self, save_path):
        with open(save_path, 'w') as f:
            f.write('\n'.join(self.labels))

    def encode(self, labels):
        return self.__getitem__(labels)

    def decode(self, labels):
        return self.__getitem__(labels)

    @classmethod
    def load_dict(cls, load_path, **kwargs):
        with open(load_path, 'r') as f:
            labels = f.read().strip('\n').split('\n')

        return cls(labels, **kwargs)
          
