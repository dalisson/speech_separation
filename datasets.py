from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import numpy as np
import math

class AudioSeparationDataset(Dataset):
    
    def __init__(self, data):
        super(AudioSeparationDataset, self).__init__()
        self.data = data
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, i):
        row = self.data.loc[i]
        x, _ = torchaudio.load(row.x).unsqueeze(0)
        y0, _ = torchaudio.load(row.y1)
        y1, _ = torchaudio.load(row.y2)
        y = torch.cat((y0, y1), dim=0)
        return x, y
    
class SeparationDatasets(object):
    
    def __init__(self, csv_file, seed=42):
        np.random.seed(seed)
        self.d = pd.read_csv(csv_file)
        train, test = self.build_separation()
        train_dataset, test_dataset = AudioSeparationDataset(self.d.loc[train]), AudioSeparationDataset(self.d.loc[test])
        self.train_dl = DataLoader(train_dataset, batch_size=1)
        self.valid_dl = DataLoader(test_dataset, batch_size=1)
        
    
    def build_separation(self):
        total_samples = self.d.shape[0]
        test_size = math.floor(total_samples*0.2)
        train_size = total_samples - test_size
        seq = np.arange(0, total_samples)
        np.random.shuffle(seq)
        train_samples = seq[:train_size]
        test_samples = seq[train_size:]
        return train_samples, test_samples