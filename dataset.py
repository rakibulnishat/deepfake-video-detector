"""
dataset.py — DFDC Dataset Loader
Loads videos from Kaggle-mounted DFDC competition data.
Extracts faces per frame using MTCNN, returns tensors for training.
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# MTCNN for face detection
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("[WARNING] facenet-pytorch not installed. Using center crop fallback.")


# ─── AUGMENTATION TRANSFORMS ─────────────────────────────────────────────────

def get_train_transforms(face_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((face_size, face_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(face_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((face_size, face_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ─── FACE EXTRACTOR ──────────────────────────────────────────────────────────

class FaceExtractor:
    """
    Extracts face crops from video frames.
    Uses MTCNN if available, otherwise falls back to center crop.
    """

    def __init__(self, face_size: int = 224, device: str = "cpu"):
        self.face_size = face_size
        self.device = device
        if MTCNN_AVAILABLE:
            self.detector = MTCNN(
                image_size=face_size,
                margin=20,
                keep_all=False,       # Only the most prominent face
                post_process=False,   # Return raw PIL images
                device=device,
                select_largest=True,  # Take the largest face if multiple
            )
        else:
            self.detector = None

    def extract_from_frame(self, frame_bgr: np.ndarray) -> Optional[Image.Image]:
        """
        Given a BGR numpy frame (from cv2), return a PIL face crop or None.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        if self.detector is not None:
            try:
                face = self.detector(pil_img)
                if face is not None:
                    # MTCNN returns a tensor; convert back to PIL
                    face_np = face.permute(1, 2, 0).numpy()
                    face_np = np.clip(face_np, 0, 255).astype(np.uint8)
                    return Image.fromarray(face_np)
            except Exception:
                pass  # Fall through to center crop

        # Fallback: center crop
        w, h = pil_img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        pil_img = pil_img.crop((left, top, left + min_dim, top + min_dim))
        return pil_img.resize((self.face_size, self.face_size))


# ─── VIDEO FRAME SAMPLER ─────────────────────────────────────────────────────

def sample_frames_from_video(
    video_path: str,
    num_frames: int = 8,
    face_extractor: Optional[FaceExtractor] = None,
    face_size: int = 224,
) -> Optional[List[np.ndarray]]:
    """
    Opens a video, uniformly samples `num_frames` frames,
    extracts faces, and returns a list of PIL images.
    Returns None if video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        indices = list(range(total_frames))
    else:
        # Uniform sampling across the video
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        if face_extractor is not None:
            face = face_extractor.extract_from_frame(frame)
            if face is not None:
                frames.append(face)
        else:
            # No face extraction — just resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb).resize((face_size, face_size))
            frames.append(pil_img)

    cap.release()

    # Pad if not enough frames were extracted
    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])  # Repeat last frame

    return frames if len(frames) > 0 else None


# ─── DFDC DATASET ────────────────────────────────────────────────────────────

class DFDCDataset(Dataset):
    """
    Dataset for the Deepfake Detection Challenge (DFDC).

    Reads metadata.json from each part folder, collects video paths
    and binary labels (0=REAL, 1=FAKE), samples frames, and returns
    a tensor of shape [num_frames, C, H, W].

    Args:
        root_dir:       Path to DFDC train folder.
                        e.g. "/kaggle/input/deepfake-detection-challenge/train"
        num_parts:      How many part folders to load (1–50).
        num_frames:     Frames to sample per video.
        face_size:      Face crop resolution.
        transform:      torchvision transform applied to each frame.
        split:          "train" or "val".
        val_split:      Fraction of data reserved for validation.
        max_per_part:   Max videos per part (None = all). Useful for quick tests.
        seed:           Random seed for reproducible splits.
        device:         Device for MTCNN ("cuda" or "cpu").
    """

    def __init__(
        self,
        root_dir: str,
        num_parts: int = 5,
        num_frames: int = 8,
        face_size: int = 224,
        transform=None,
        split: str = "train",
        val_split: float = 0.15,
        max_per_part: Optional[int] = None,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.face_size = face_size
        self.transform = transform
        self.split = split

        self.face_extractor = FaceExtractor(face_size=face_size, device=device)

        # ── Collect all samples ──
        self.samples: List[Tuple[str, int]] = []   # (video_path, label)
        self._collect_samples(num_parts, max_per_part)

        # ── Train/val split ──
        random.seed(seed)
        random.shuffle(self.samples)
        n_val = int(len(self.samples) * val_split)
        if split == "val":
            self.samples = self.samples[:n_val]
        else:
            self.samples = self.samples[n_val:]

        # ── Class balance info ──
        labels = [s[1] for s in self.samples]
        n_fake = sum(labels)
        n_real = len(labels) - n_fake
        print(f"[DFDCDataset] {split}: {len(self.samples)} videos "
              f"({n_real} REAL, {n_fake} FAKE)")

    def _collect_samples(self, num_parts: int, max_per_part: Optional[int]):
        """Walk through part folders and collect (video_path, label) pairs."""
        part_dirs = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir() and d.name.startswith("dfdc_train_part_")
        ])[:num_parts]

        if len(part_dirs) == 0:
            raise FileNotFoundError(
                f"No 'dfdc_train_part_*' folders found in {self.root_dir}. "
                "Make sure the DFDC dataset is attached to your Kaggle notebook."
            )

        for part_dir in part_dirs:
            meta_path = part_dir / "metadata.json"
            if not meta_path.exists():
                print(f"[WARNING] No metadata.json in {part_dir}, skipping.")
                continue

            with open(meta_path, "r") as f:
                metadata: Dict = json.load(f)

            part_samples = []
            for filename, info in metadata.items():
                video_path = part_dir / filename
                if not video_path.exists():
                    continue
                label = 1 if info.get("label", "REAL") == "FAKE" else 0
                part_samples.append((str(video_path), label))

            if max_per_part is not None:
                random.shuffle(part_samples)
                part_samples = part_samples[:max_per_part]

            self.samples.extend(part_samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        video_path, label = self.samples[idx]

        frames = sample_frames_from_video(
            video_path,
            num_frames=self.num_frames,
            face_extractor=self.face_extractor,
            face_size=self.face_size,
        )

        # If video failed to load, return a black tensor
        if frames is None or len(frames) == 0:
            dummy = torch.zeros(self.num_frames, 3, self.face_size, self.face_size)
            return {"frames": dummy, "label": torch.tensor(label, dtype=torch.float32),
                    "path": video_path}

        if self.transform is not None:
            frames = [self.transform(f) for f in frames]
        else:
            to_tensor = transforms.Compose([
                transforms.Resize((self.face_size, self.face_size)),
                transforms.ToTensor(),
            ])
            frames = [to_tensor(f) for f in frames]

        # Stack: [num_frames, C, H, W]
        frames_tensor = torch.stack(frames, dim=0)

        return {
            "frames": frames_tensor,                          # [T, C, H, W]
            "label": torch.tensor(label, dtype=torch.float32),
            "path": video_path,
        }


# ─── DATALOADER FACTORY ──────────────────────────────────────────────────────

def build_dataloaders(cfg: dict, device: str = "cpu") -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from config dict.
    """
    d = cfg["data"]
    t = cfg["training"]

    train_ds = DFDCDataset(
        root_dir=d["root"],
        num_parts=d["num_parts"],
        num_frames=d["frames_per_video"],
        face_size=d["face_size"],
        transform=get_train_transforms(d["face_size"]),
        split="train",
        val_split=d["val_split"],
        max_per_part=d.get("max_per_part"),
        device=device,
    )

    val_ds = DFDCDataset(
        root_dir=d["root"],
        num_parts=d["num_parts"],
        num_frames=d["frames_per_video"],
        face_size=d["face_size"],
        transform=get_val_transforms(d["face_size"]),
        split="val",
        val_split=d["val_split"],
        max_per_part=d.get("max_per_part"),
        device=device,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=t["batch_size"],
        shuffle=True,
        num_workers=d["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=t["batch_size"],
        shuffle=False,
        num_workers=d["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader
