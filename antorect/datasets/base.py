import os
import random
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pydicom
import torch


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def load_dicom_as_array(dicom_path):
    ds = pydicom.dcmread(dicom_path, force=True)
    img_array = ds.pixel_array.astype(np.float32)
        
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
         img_array = img_array * ds.RescaleSlope + ds.RescaleIntercept
        
    return img_array, ds


class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=['ima', 'dcm', 'dicom'],
        augment_flip=False,
        convert_image_to=None,
        condition=0,
        equalizeHist=False,
        crop_patch=True,
        sample=False,
        hu_min=-1024,
        hu_max=3072
    ):
        super().__init__()
        self.equalizeHist = equalizeHist
        self.exts = exts
        self.augment_flip = augment_flip
        self.condition = condition
        self.crop_patch = crop_patch
        self.sample = sample
        self.hu_min = hu_min
        self.hu_max = hu_max
        
        self.dicom_metadata = {}
        
        if condition == 1:
            self.gt = self.load_flist(folder[0])
            self.input = self.load_flist(folder[1])
        elif condition == 0:
            self.paths = self.load_flist(folder)
        elif condition == 2:
            self.gt = self.load_flist(folder[0])
            self.input = self.load_flist(folder[1])
            self.input_condition = self.load_flist(folder[2])

        self.image_size = image_size
        self.convert_image_to = convert_image_to

    def __len__(self):
        if self.condition:
            return len(self.input)
        else:
            return len(self.paths)

    def _dicom_to_pil(self, dicom_path):
        img_array, ds = load_dicom_as_array(dicom_path)
        self.dicom_metadata[dicom_path] = ds
        
        img_array = np.clip(img_array, self.hu_min, self.hu_max)
        img_array = (img_array - self.hu_min) / (self.hu_max - self.hu_min)
        img_array = (img_array * 255).astype(np.uint8)
        
        return Image.fromarray(img_array, mode='L')

    def __getitem__(self, index):
        if self.condition == 1:
            img0 = self._dicom_to_pil(self.gt[index])
            img1 = self._dicom_to_pil(self.input[index])
            
            img0_np = np.array(img0, dtype=np.float32) / 255.0
            img1_np = np.array(img1, dtype=np.float32) / 255.0
            
            if self.augment_flip and random.random() > 0.5:
                img0_np = np.fliplr(img0_np).copy()
                img1_np = np.fliplr(img1_np).copy()
            
            img0_t = torch.from_numpy(img0_np).unsqueeze(0)
            img1_t = torch.from_numpy(img1_np).unsqueeze(0)
            
            return [img0_t, img1_t]
            
        elif self.condition == 0:
            img = self._dicom_to_pil(self.paths[index])
            img_np = np.array(img, dtype=np.float32) / 255.0
            
            if self.augment_flip and random.random() > 0.5:
                img_np = np.fliplr(img_np).copy()
            
            img_t = torch.from_numpy(img_np).unsqueeze(0)
            return img_t
            
        elif self.condition == 2:
            img0 = self._dicom_to_pil(self.gt[index])
            img1 = self._dicom_to_pil(self.input[index])
            img2 = self._dicom_to_pil(self.input_condition[index])
            
            img0_np = np.array(img0, dtype=np.float32) / 255.0
            img1_np = np.array(img1, dtype=np.float32) / 255.0
            img2_np = np.array(img2, dtype=np.float32) / 255.0
            
            if self.augment_flip and random.random() > 0.5:
                img0_np = np.fliplr(img0_np).copy()
                img1_np = np.fliplr(img1_np).copy()
                img2_np = np.fliplr(img2_np).copy()
            
            img0_t = torch.from_numpy(img0_np).unsqueeze(0)
            img1_t = torch.from_numpy(img1_np).unsqueeze(0)
            img2_t = torch.from_numpy(img2_np).unsqueeze(0)
            
            return [img0_t, img1_t, img2_t]

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        if isinstance(flist, str):
            if os.path.isdir(flist):
                return [str(p) for ext in self.exts for p in Path(flist).glob(f'**/*.{ext}')]

            if os.path.isfile(flist):
                if flist.endswith('.flist') or flist.endswith('.txt'):
                    try:
                        with open(flist, 'r', encoding='utf-8') as f:
                            paths = [line.strip() for line in f if line.strip()]
                        valid_paths = []
                        for path in paths:
                            if os.path.exists(path):
                                valid_paths.append(path)
                            else:
                                print(f"Warning: Path does not exist: {path}")
                        return valid_paths
                    except Exception as e:
                        print(f"Failed to read file list: {flist}, error: {e}")
                        return []
                else:
                    return [flist]
        return []

    def load_name(self, index, sub_dir=False):
        if self.condition:
            name = self.input[index]
            if sub_dir == 0:
                return os.path.basename(name)
            elif sub_dir == 1:
                path = os.path.dirname(name)
                sub_dir = (path.split("/"))[-1]
                return sub_dir + "_" + os.path.basename(name)

    def get_pad_size(self, index, block_size=8):
        return [0, 0]
    
    def get_dicom_metadata(self, index):
        if self.condition:
            path = self.input[index]
        else:
            path = self.paths[index]
        return self.dicom_metadata.get(path, None)