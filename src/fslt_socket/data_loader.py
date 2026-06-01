from __future__ import annotations

import os
from glob import glob
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


LESION_TYPES = {
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
    "bkl": "Benign keratosis-like lesions",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses",
    "vasc": "Vascular lesions",
    "df": "Dermatofibroma",
}


class SkinDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None) -> None:
        self.frame = frame
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        img_path = self.frame.iloc[index]["path"]
        label = int(self.frame.iloc[index]["target"])
        image = Image.open(img_path).convert("RGB").resize((64, 64))
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def load_ham10000_frame(metadata_path: Path, image_glob: str) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing HAM10000 metadata at {metadata_path}. "
            "Download the dataset locally; do not commit it to GitHub."
        )

    frame = pd.read_csv(metadata_path)
    image_paths = {
        os.path.splitext(os.path.basename(path))[0]: path for path in glob(image_glob)
    }
    if not image_paths:
        raise FileNotFoundError(
            f"No HAM10000 image files matched {image_glob!r}. "
            "Expected data/HAM10000_images_part_*/ISIC_*.jpg."
        )

    frame["path"] = frame["image_id"].map(image_paths.get)
    frame = frame.dropna(subset=["path"]).reset_index(drop=True)
    frame["cell_type"] = frame["dx"].map(LESION_TYPES.get)
    frame["target"] = pd.Categorical(frame["cell_type"]).codes
    return frame


def build_client_loaders(
    client_id: int,
    num_clients: int,
    batch_size: int,
    metadata_path: Path,
    image_glob: str,
) -> tuple[DataLoader, DataLoader]:
    frame = load_ham10000_frame(metadata_path, image_glob)
    frame = frame.sample(frac=1, random_state=42).reset_index(drop=True)
    train_frame, test_frame = train_test_split(frame, test_size=0.2, random_state=42)

    train_split = _client_slice(train_frame, client_id, num_clients)
    test_split = _client_slice(test_frame, client_id, num_clients)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return (
        DataLoader(SkinDataset(train_split, transform), batch_size=batch_size, shuffle=True),
        DataLoader(SkinDataset(test_split, transform), batch_size=batch_size, shuffle=False),
    )


def _client_slice(frame: pd.DataFrame, client_id: int, num_clients: int) -> pd.DataFrame:
    split_size = len(frame) // num_clients
    start = client_id * split_size
    end = len(frame) if client_id == num_clients - 1 else start + split_size
    return frame.iloc[start:end].reset_index(drop=True)
