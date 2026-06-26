# utils/ct_augmentation.py
import numpy as np
import random


class CTDataAugmentation:
    def __init__(self):
        self.noise_std_range = [0.01, 0.03]
        self.contrast_range = [0.9, 1.1]
        self.brightness_range = [-0.05, 0.05]
    
    def create_two_views(self, batch_images):
        batch_size = batch_images.shape[0]
        view1 = np.zeros_like(batch_images)
        view2 = np.zeros_like(batch_images)
        
        for i in range(batch_size):
            img = batch_images[i, 0]
            view1[i, 0] = self._augment(img, mode='light')
            view2[i, 0] = self._augment(img, mode='strong')
        
        return view1, view2
    
    def _augment(self, image, mode='medium'):
        if mode == 'light':
            noise_std = random.uniform(0.005, 0.015)
            contrast = random.uniform(0.95, 1.05)
            brightness = random.uniform(-0.03, 0.03)
        else:
            noise_std = random.uniform(0.02, 0.03)
            contrast = random.uniform(0.9, 1.1)
            brightness = random.uniform(-0.08, 0.08)
        
        augmented = image.copy()
        
        mean_val = np.mean(augmented)
        augmented = (augmented - mean_val) * contrast + mean_val
        augmented = augmented + brightness
        
        augmented = augmented + np.random.normal(0, noise_std, augmented.shape)
        
        if random.random() > 0.5:
            augmented = np.fliplr(augmented)
        
        return np.clip(augmented, 0.0, 1.0)