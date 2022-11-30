from typing import Tuple
import tempfile
from pathlib import Path

import numpy as np
import svgwrite
import cairosvg
from PIL import Image
import tqdm

import torch
import torchvision
from torchvision import transforms as T


class SVGDataset(torchvision.datasets.VisionDataset):

    def __init__(self,
                 data_file,
                 size=50000,
                 image_size=16,
                 patch_size=4,
                 ) -> None:
        super().__init__(None)
        self.data_file = Path(data_file)
        self.size = size
        self.image_size = image_size
        self.patch_size = patch_size

        if not self.data_file.exists():
            self.generate()
        else:
            self.load()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.12, 0.12, 0.12],
                        std=[0.3, 0.3, 0.3]),
        ])
        self.target_transform = T.ToTensor()

    def generate(self):
        data_path = self.data_file.parent
        data_path.mkdir(parents=True, exist_ok=True)

        generator = OverlappingShapes(image_size=self.image_size)
        samples = []
        for _ in tqdm.trange(self.size, desc=f'Generating {self.data_file.stem}'):
            samples.append(generator())

        self.samples = np.stack(samples)
        np.save(self.data_file, self.samples)

    def load(self):
        self.samples = np.load(self.data_file)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target = self.samples[index]
        # convert to binary class per pixel, keep 3 dimensions
        target = (target > 0).any(axis=2, keepdims=True).astype(np.float32)

        img = self.transform(img)
        target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return self.size


class GenSVGDataset(torchvision.datasets.VisionDataset):

    def __init__(self,
                 size=50000,
                 image_size=16,
                 patch_size=4,
                 ) -> None:
        super().__init__(None)
        self.size = size
        self.image_size = image_size
        self.patch_size = patch_size

        self.generator = OverlappingShapes(image_size=self.image_size)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.12, 0.12, 0.12],
                        std=[0.3, 0.3, 0.3]),
        ])
        self.target_transform = T.ToTensor()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target = self.generator()
        # convert to binary class per pixel, keep 3 dimensions
        target = (target > 0).any(axis=2, keepdims=True).astype(np.float32)

        img = self.transform(img)
        # target = self.target_transform(target)
        target = torch.randint(0, 9, size=(1,)).item()
        return img, target

    def __len__(self) -> int:
        return self.size


class SVG:

    def __init__(self, image_size=16) -> np.ndarray:
        self.image_size = image_size

    def __call__(self):
        self.drawing = svgwrite.Drawing(
            'test.png',  # not used anywhere
            size=(f'{self.image_size}px', f'{self.image_size}px'),
            profile='tiny'
        )

    def add_rect(self,
                 xc: float,
                 yc: float,
                 w: float,
                 h: float,
                 color: str
                 ) -> None:
        x = xc - w // 2
        y = yc - h // 2
        element = self.drawing.rect(
            insert=(int(x), int(y)),
            size=(int(w), int(h)),
            fill=color
        )
        self.drawing.add(element)

    def add_ellipse(self: svgwrite.Drawing,
                    xc: float,
                    yc: float,
                    w: float,
                    h: float,
                    color: str
                    ) -> None:
        rx = w // 2
        ry = h // 2
        element = self.drawing.ellipse(
            center=(int(xc), int(yc)),
            r=(int(rx), int(ry)),
            fill=color
        )
        self.drawing.add(element)

    def to_img(self: svgwrite.Drawing):
        f = self.drawing.tostring()
        with tempfile.TemporaryFile() as fp:
            cairosvg.svg2png(bytestring=f, write_to=fp)
            img = np.array(Image.open(fp)).astype(np.uint8)
        return img


class OverlappingShapes(SVG):

    COLORS = ['white', 'gray', 'red', 'yellow', 'green', 'cyan', 'blue', 'magenta']

    def __init__(self, image_size=16, max_iter=100) -> np.ndarray:
        super().__init__(image_size=image_size)
        self.max_iter = max_iter

    def __call__(self):
        super().__call__()

        bckg = self.drawing.rect(
            insert=(0, 0),
            size=(self.image_size, self.image_size),
            fill='black'
        )
        self.drawing.add(bckg)

        colors = np.random.choice(self.COLORS, size=2, replace=False)

        # background
        aspect = 2 ** np.random.uniform(-1, 1)  # will be between .5 and 2
        w0 = np.random.random_integers(self.image_size // 3, 2 * self.image_size // 3)
        h0 = max(aspect * w0, self.image_size // 3)
        xc0, yc0 = np.random.random_integers(0, self.image_size, size=2)

        func = np.random.choice([self.add_rect, self.add_ellipse])
        func(xc0, yc0, w0, h0, colors[0])
        img_bg = self.to_img()

        # foreground
        aspect = 2 ** np.random.uniform(-1, 1)  # will be between .5 and 2
        w1 = np.random.random_integers(min(self.image_size // 3, w0 - 1), min(self.image_size // 2, w0))
        h1 = min(max(aspect * w1, self.image_size // 3), h0)

        for _ in range(self.max_iter):
            dim = np.random.choice(['x', 'y'])
            if dim == 'x':
                xc1 = xc0 + np.random.choice([-1, 1]) * w0 // 2
                yc1 = np.random.random_integers(yc0 - h0 // 2, yc0 + h0 // 2)
            else:
                yc1 = yc0 + np.random.choice([-1, 1]) * h0 // 2
                xc1 = np.random.random_integers(xc0 - w0 // 2, xc0 + w0 // 2)
            if xc1 >= 0 and xc1 < self.image_size and yc1 >= 0 and yc1 < self.image_size:
                break

        func = np.random.choice([self.add_rect, self.add_ellipse])
        func(xc1, yc1, w1, h1, colors[1])

        img = self.to_img()

        return img, img_bg
