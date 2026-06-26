# datasets/hasc_dataset.py
import os
import glob
import random
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset


class HASCDataset(Dataset):
    def __init__(self, data_root, train_patients, slice_num=4, patch_num=8, 
                 patch_size=64, u_water=0.0205):
        super().__init__()
        self.data_root = data_root
        self.slice_num = slice_num
        self.patch_num = patch_num
        self.patch_size = patch_size
        self.u_water = u_water
        self.pairs = []
        
        self._collect_pairs(train_patients)
    
    def _collect_pairs(self, patients):
        for patient in patients:
            ld_dir = os.path.join(self.data_root, 'quarter_1mm', patient, 'quarter_1mm')
            nd_dir = os.path.join(self.data_root, 'full_1mm', patient, 'full_1mm')
            
            if not os.path.exists(ld_dir) or not os.path.exists(nd_dir):
                continue
            
            ld_files = sorted(glob.glob(os.path.join(ld_dir, '*.IMA')))
            nd_files = sorted(glob.glob(os.path.join(nd_dir, '*.IMA')))
            
            min_len = min(len(ld_files), len(nd_files))
            for i in range(min_len):
                self.pairs.append((ld_files[i], nd_files[i]))
        
        print(f"HASC Dataset: {len(self.pairs)} pairs of slices")
    
    def _load_dicom(self, path):
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img = ((img - 1000.) / 1000.0) * self.u_water + self.u_water
        img[img < 0] = 0
        
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        
        return img
    
    def _extract_patches(self, image):
        h, w = image.shape
        patches = []
        for _ in range(self.patch_num):
            i = random.randint(0, h - self.patch_size)
            j = random.randint(0, w - self.patch_size)
            patch = image[i:i+self.patch_size, j:j+self.patch_size]
            patches.append(patch)
        return np.stack(patches, axis=0)
    
    def __getitem__(self, idx):
        selected = random.sample(range(len(self.pairs)), self.slice_num)
        
        ld_slices, nd_slices = [], []
        for slice_idx in selected:
            ld_path, nd_path = self.pairs[slice_idx]
            ld_img = self._load_dicom(ld_path)
            nd_img = self._load_dicom(nd_path)
            
            ld_patches = self._extract_patches(ld_img)
            nd_patches = self._extract_patches(nd_img)
            
            ld_slices.append(ld_patches)
            nd_slices.append(nd_patches)
        
        ld_tensor = torch.from_numpy(np.stack(ld_slices, axis=0)).float().unsqueeze(2)
        nd_tensor = torch.from_numpy(np.stack(nd_slices, axis=0)).float().unsqueeze(2)
        
        return ld_tensor, nd_tensor
    
    def __len__(self):
        return max(1, len(self.pairs) // self.slice_num)