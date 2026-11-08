"""Content/style dataset over two image folders (e.g. MS-COCO + WikiArt).

Each item returns one content crop and one randomly paired style crop, both in [0, 1].
Images are resized so the short side is `resize`, then randomly cropped to `img_size`
(the standard arbitrary-style-transfer preprocessing).
"""
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import functional as TF

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _list_images(root):
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(_EXTS):
                paths.append(os.path.join(dirpath, f))
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    return sorted(paths)


class SeededEpochSampler(Sampler):
    """A shuffled index order that is a pure function of (seed, epoch).

    The draw does not touch the global generators, so two runs with the same seed see
    the same order even when their models consume different amounts of randomness at
    initialisation. `start` skips the items a resumed run has already used.
    """

    def __init__(self, n, seed):
        self.n, self.seed, self.epoch, self.start = n, int(seed), 0, 0

    def set_position(self, epoch, start=0):
        self.epoch, self.start = epoch, start

    def __len__(self):
        return self.n - self.start

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 1000003 + self.epoch)
        return iter(torch.randperm(self.n, generator=g)[self.start:].tolist())


class ContentStyleDataset(Dataset):
    def __init__(self, content_dir, style_dir, img_size=256, resize=512, seed=None):
        self.content_paths = _list_images(content_dir)
        self.style_paths = _list_images(style_dir)
        self.img_size, self.seed, self.epoch = img_size, seed, 0
        self.resize = transforms.Resize(resize)
        self.tf = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomCrop(img_size, pad_if_needed=True, padding_mode="reflect"),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.content_paths)

    def _load(self, path):
        img = Image.open(path).convert("RGB")
        return self.tf(img)

    def _load_seeded(self, path, rng):
        img = self.resize(Image.open(path).convert("RGB"))
        w, h = img.size
        if w < self.img_size or h < self.img_size:
            img = TF.pad(img, [max(self.img_size - w, 0), max(self.img_size - h, 0)],
                         padding_mode="reflect")
            w, h = img.size
        top = rng.randint(0, h - self.img_size)
        left = rng.randint(0, w - self.img_size)
        return TF.to_tensor(TF.crop(img, top, left, self.img_size, self.img_size))

    def __getitem__(self, idx):
        if self.seed is not None:
            # The style choice and both crops depend only on (seed, epoch, idx), so the
            # sequence of training examples is the same for every model under a seed and
            # does not change with the number of loader workers.
            rng = random.Random(f"{self.seed}:{self.epoch}:{idx}")
            content = self._load_seeded(self.content_paths[idx], rng)
            style = self._load_seeded(rng.choice(self.style_paths), rng)
            return content, style
        content = self._load(self.content_paths[idx])
        style = self._load(random.choice(self.style_paths))
        return content, style
