# datasets/simsiam_dataset.py
import os
import glob
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset


class SimSiamDataset(Dataset):
    def __init__(self, data_root, patient_list, dose_types=['quarter_1mm', 'full_1mm'],
                 u_water=0.0205):
        super().__init__()
        self.u_water = u_water
        self.file_paths = []
        
        for patient in patient_list:
            for dose_type in dose_types:
                dose_path = os.path.join(data_root, dose_type, patient, dose_type)
                if not os.path.exists(dose_path):
                    continue
                files = glob.glob(os.path.join(dose_path, '*.IMA'))
                self.file_paths.extend(files)
        
        print(f"SimSiam Dataset: {len(self.file_paths)} images")
    
    def _load_dicom(self, path):
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img = ((img - 1000.) / 1000.0) * self.u_water + self.u_water
        img[img < 0] = 0
        
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        
        return img
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img = self._load_dicom(self.file_paths[idx])
        img = torch.from_numpy(img).float().unsqueeze(0)
        return img