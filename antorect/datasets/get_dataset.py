# get_dataset.py
from datasets.base import Dataset

def dataset(folder,
            image_size,
            exts=['ima', 'dcm', 'dicom'],
            augment_flip=False,
            convert_image_to=None,
            condition=0,
            equalizeHist=False,
            crop_patch=True,
            sample=False,
            generation=False):
    
    return Dataset(
        folder=folder,
        image_size=image_size,
        exts=exts,
        augment_flip=augment_flip,
        convert_image_to=convert_image_to,
        condition=condition,
        equalizeHist=equalizeHist,
        crop_patch=crop_patch,
        sample=sample
    )