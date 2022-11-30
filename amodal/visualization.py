from pathlib import Path
import matplotlib.pyplot as plt
import torch
from amodal import train, dataset


model = train.Model.load_from_checkpoint(train.DATA_PATH / 'last.ckpt')
data = dataset.SVGDataset(train.DATA_PATH / 'val.npy')
generator = dataset.OverlappingShapes(image_size=16)

fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(6, 2))
for ax1, ax2 in axes.T:
    img, _ = generator()
    x = data.transform(img).unsqueeze(0)
    x = model.model.img_to_patch(x)

    out = model(x)
    y_hat = (model.model
             .patch_to_img(torch.sigmoid(out))
             .repeat([1, 3, 1, 1])
             .permute(0, 2, 3, 1)
             .detach()
             .numpy()
             [0]
             )

    ax1.imshow(img)
    ax2.imshow(y_hat)

plt.show()
