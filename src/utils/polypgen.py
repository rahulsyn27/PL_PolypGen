import os
from pathlib import Path
from typing import Optional, Callable, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class PolypGenDataset(Dataset):
    """
    PolypGen Dataset supporting:
    - 6 centers (C1-C6) for multi-center feature shift (H2)
    - Combined pre-extracted numpy arrays (data.npy, targets.npy)
    - On-the-fly loading of raw colonoscopy frames and segmentation masks
    """

    def __init__(
        self,
        root_dir: str,
        center_id: Optional[int] = None,
        transform: Optional[Callable] = None,
    ):
        self.root_dir = str(root_dir)
        self.center_id = center_id
        self.transform = transform
        self.classes = [0, 1]

        data_file = os.path.join(self.root_dir, "data.npy")
        targets_file = os.path.join(self.root_dir, "targets.npy")

        # Case 1: Pre-extracted combined numpy files exist and no single center requested
        if os.path.exists(data_file) and os.path.exists(targets_file) and center_id is None:
            raw_data = np.load(data_file)
            # Ensure shape is (N, C, H, W) and normalized float
            if raw_data.ndim == 4 and raw_data.shape[-1] == 3:
                # NHWC -> NCHW
                raw_data = np.transpose(raw_data, (0, 3, 1, 2))
            self.data = torch.from_numpy(raw_data).float()
            if self.data.max() > 1.0:
                self.data = self.data / 255.0
            self.targets = torch.from_numpy(np.load(targets_file)).long()
            self._is_combined = True

        # Case 2: Specific center requested (C1 through C6)
        elif center_id is not None:
            center_folder = f"C{center_id}"
            center_path = os.path.join(self.root_dir, center_folder)
            images_dir = os.path.join(center_path, "images")
            masks_dir = os.path.join(center_path, "masks")

            # Official PolypGen release layout: data_C{id}/images_C{id}, data_C{id}/masks_C{id}
            official_path = os.path.join(self.root_dir, f"data_{center_folder}")
            if not os.path.exists(images_dir) and os.path.exists(official_path):
                images_dir = os.path.join(official_path, f"images_{center_folder}")
                masks_dir = os.path.join(official_path, f"masks_{center_folder}")

            # Fallback if images/masks are directly inside center_folder
            if not os.path.exists(images_dir):
                images_dir = center_path
                masks_dir = os.path.join(self.root_dir, f"{center_folder}_masks")

            self.image_paths = []
            if os.path.exists(images_dir):
                for fname in sorted(os.listdir(images_dir)):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                        img_path = os.path.join(images_dir, fname)
                        mask_path = ""
                        if os.path.exists(masks_dir):
                            # Masks are named either identically or as <stem>_mask.<ext>
                            stem, ext = os.path.splitext(fname)
                            for cand in (fname, f"{stem}_mask{ext}", f"{stem}_mask.jpg", f"{stem}_mask.png"):
                                if os.path.exists(os.path.join(masks_dir, cand)):
                                    mask_path = os.path.join(masks_dir, cand)
                                    break
                        self.image_paths.append((img_path, mask_path))

            # If directory didn't have raw images, check if per-center npy exists (data_c{id}.npy)
            per_center_data = os.path.join(self.root_dir, f"data_c{center_id}.npy")
            per_center_targets = os.path.join(self.root_dir, f"targets_c{center_id}.npy")
            if os.path.exists(per_center_data) and os.path.exists(per_center_targets):
                raw_data = np.load(per_center_data)
                if raw_data.ndim == 4 and raw_data.shape[-1] == 3:
                    raw_data = np.transpose(raw_data, (0, 3, 1, 2))
                self.data = torch.from_numpy(raw_data).float()
                if self.data.max() > 1.0:
                    self.data = self.data / 255.0
                self.targets = torch.from_numpy(np.load(per_center_targets)).long()
                self._is_combined = True
            elif not self.image_paths:
                # No real data found for this center — raise so experiments are never silently
                # corrupted by dummy tensors on a production server.
                raise FileNotFoundError(
                    f"No image data found for center C{center_id}.\n"
                    f"  Checked image dir  : {images_dir}\n"
                    f"  Checked npy files  : {per_center_data}, {per_center_targets}\n"
                    f"Please download the PolypGen dataset and place it under: {self.root_dir}"
                )
            else:
                self._is_combined = False
        else:
            # center_id=None and no combined npy files: pool raw images from all centers C1-C6
            self.image_paths = []
            for cid in range(1, 7):
                self.image_paths.extend(PolypGenDataset(self.root_dir, center_id=cid).image_paths)
            self._is_combined = False
            self.targets = torch.tensor([self._load(i)[1] for i in range(len(self.image_paths))], dtype=torch.long)

    def __len__(self) -> int:
        if self._is_combined:
            return len(self.targets)
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._is_combined:
            image = self.data[idx]
            label = self.targets[idx]
            if self.transform:
                image = self.transform(image)
            return image, label

        image_u8, label = self._load(idx)
        # HWC uint8 -> CHW float in [0, 1]
        image_tensor = torch.from_numpy(image_u8.transpose((2, 0, 1))).float().div_(255.0)

        label_tensor = torch.tensor(label, dtype=torch.long)
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor, label_tensor

    def _load(self, idx: int) -> Tuple[np.ndarray, int]:
        """Decode + resize an image and derive its label from the mask, cached in memory as uint8.
        JPEG decoding of full-resolution frames otherwise dominates epoch time and starves the GPU."""
        cache = self.__dict__.setdefault("_cache", {})
        if idx in cache:
            return cache[idx]

        img_path, mask_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image.resize((224, 224)), dtype=np.uint8)

        label = 0
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            if np.array(mask).max() > 0:
                label = 1

        cache[idx] = (image_np, label)
        return cache[idx]
