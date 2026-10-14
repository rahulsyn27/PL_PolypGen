import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np


class PolypGenDataset(Dataset):
    def __init__(self, root_dir, center_id=None, transform=None):
        self.root_dir = root_dir
        self.center_id = center_id
        self.transform = transform

        # Check if combined data files exist (for framework's get_dataset)
        data_file = os.path.join(root_dir, "data.npy")
        targets_file = os.path.join(root_dir, "targets.npy")

        if os.path.exists(data_file) and os.path.exists(targets_file) and center_id is None:
            # Load combined dataset for federated learning
            self.data = torch.from_numpy(np.load(data_file)).float()
            self.targets = torch.from_numpy(np.load(targets_file)).long()
            self.classes = [0, 1]
            self._is_combined = True
        elif center_id is not None:
            # Load single center dataset (for data generation)
            center_folder = f"C{center_id}"
            images_dir = os.path.join(root_dir, center_folder, "images")
            masks_dir = os.path.join(root_dir, center_folder, "masks")

            if not os.path.exists(images_dir):
                raise FileNotFoundError(f"Images directory not found: {images_dir}")

            self.image_paths = []
            for fname in sorted(os.listdir(images_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    img_path = os.path.join(images_dir, fname)
                    mask_path = os.path.join(masks_dir, fname)
                    self.image_paths.append((img_path, mask_path))

            if not self.image_paths:
                raise RuntimeError(f"No images found in {images_dir}")
            self._is_combined = False
        else:
            raise ValueError("Either center_id must be provided or combined data files must exist")

    def __len__(self):
        if self._is_combined:
            return len(self.targets)
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self._is_combined:
            image = self.data[idx]
            label = self.targets[idx]
            if self.transform:
                image = self.transform(image)
            return image, label

        img_path, mask_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

        label = 0
        if os.path.exists(mask_path):
            try:
                mask = Image.open(mask_path).convert("L")
                mask_array = np.array(mask)
                if mask_array.max() > 0:
                    label = 1
            except Exception as e:
                raise RuntimeError(f"Failed to load mask {mask_path}: {e}")
        else:
            raise FileNotFoundError(f"Mask not found for image {img_path}: {mask_path}")

        return image, torch.tensor(label, dtype=torch.long)