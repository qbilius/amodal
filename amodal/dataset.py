from typing import Tuple, Union
import tempfile

import numpy as np
import svgwrite
import cairosvg
from PIL import Image

import torch
import torchvision
from torchvision import transforms as T


class GenSVGDataset(torchvision.datasets.VisionDataset):

    def __init__(self,
                 size=50000,
                 image_size=32,
                 patch_size=4,
                 #  seed=0
                 ) -> None:
        super().__init__(None)
        self.size = size
        self.image_size = image_size
        self.patch_size = patch_size
        self.rng = np.random.default_rng()  # TODO: Fix seeding

        self.generator = OverlappingShapes(self.rng, image_size=self.image_size)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.88, 0.88, 0.88],
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
        target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return self.size


class SVG:

    def __init__(self,
                 rng: Union[np.random.Generator, int] = None,
                 image_size=32
                 ) -> np.ndarray:

        if rng is None:
            self.rng = np.random.default_rng()
        elif isinstance(rng, int):
            self.rng = np.random.default_rng(rng)
        elif isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            raise ValueError('rng must be')

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

    def __init__(self,
                 rng: Union[np.random.Generator, int] = None,
                 image_size=32,
                 max_iter=100,
                 ) -> np.ndarray:
        super().__init__(rng=rng, image_size=image_size)
        self.max_iter = max_iter

    def __call__(self):
        super().__call__()

        bckg = self.drawing.rect(
            insert=(0, 0),
            size=(self.image_size, self.image_size),
            fill='black'
        )
        self.drawing.add(bckg)

        colors = self.rng.choice(self.COLORS, size=2, replace=False)

        # background
        aspect = 2 ** self.rng.uniform(-1, 1)  # will be between .5 and 2
        w0 = self.rng.integers(self.image_size // 3, 2 * self.image_size // 3)
        h0 = max(aspect * w0, self.image_size // 3)
        xc0, yc0 = self.rng.integers(0, self.image_size, size=2)

        func = self.rng.choice([self.add_rect, self.add_ellipse])
        func(xc0, yc0, w0, h0, colors[0])
        img_bg = self.to_img()

        # foreground
        aspect = 2 ** self.rng.uniform(-1, 1)  # will be between .5 and 2
        w1 = self.rng.integers(min(self.image_size // 3, w0 - 1), min(self.image_size // 2, w0))
        h1 = min(max(aspect * w1, self.image_size // 3), h0)

        for _ in range(self.max_iter):
            dim = self.rng.choice(['x', 'y'])
            if dim == 'x':
                xc1 = xc0 + self.rng.choice([-1, 1]) * w0 // 2
                yc1 = self.rng.integers(yc0 - h0 // 2, yc0 + h0 // 2)
            else:
                yc1 = yc0 + self.rng.choice([-1, 1]) * h0 // 2
                xc1 = self.rng.integers(xc0 - w0 // 2, xc0 + w0 // 2)
            if xc1 >= 0 and xc1 < self.image_size and yc1 >= 0 and yc1 < self.image_size:
                break

        func = self.rng.choice([self.add_rect, self.add_ellipse])
        func(xc1, yc1, w1, h1, colors[1])

        img = self.to_img()

        return img, img_bg
