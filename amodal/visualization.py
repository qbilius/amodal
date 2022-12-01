from pathlib import Path

import fire
import torch
import tbparse
import matplotlib.pyplot as plt

from amodal import train, dataset


def plot_loss(output_path=train.OUTPUT_PATH, version=1, savefig=False):
    log_dir = Path(output_path) / f'version_{version}'
    tfevents = list(log_dir.glob('*tfevents*'))[0]
    df = tbparse.SummaryReader(str(tfevents)).scalars
    pv = (df
          .loc[df.tag.isin(['train_loss', 'val_loss'])]
          .pivot(index='step', columns='tag', values='value')
          )
    pv.train_loss.plot()
    pv.val_loss.dropna().plot()
    plt.legend()
    plt.show()

    if savefig:
        plt.savefig('results/loss.png', bbox_inches='tight', transparent=True)


def plot_results(output_path=train.OUTPUT_PATH, version=1, savefig=False, seed=None):
    model = train.Model.load_from_checkpoint(str(output_path / f'version_{version}' / 'last.ckpt'))
    data = dataset.SVGDataset(train.OUTPUT_PATH / 'val.npy', seed=seed)
    generator = dataset.OverlappingShapes(image_size=16)

    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(6, 2))
    plt.suptitle('Input image (above) and predicted amodal completion (below)')
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
        ax1.get_xaxis().set_ticks([])
        ax1.get_yaxis().set_ticks([])

        ax2.imshow(y_hat)
        ax2.get_xaxis().set_ticks([])
        ax2.get_yaxis().set_ticks([])

    plt.show()

    if savefig:
        plt.savefig('results/results.png', bbox_inches='tight', transparent=True)


if __name__ == '__main__':
    fire.Fire()
