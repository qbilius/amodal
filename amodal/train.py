from pathlib import Path
import argparse, shutil

import numpy as np
import torch, torch.utils.data
from torch.nn import functional as F
import torchvision
import pytorch_lightning as pl

import dataset, model


# worker_init_fn must be picklable so we list them here
def set_numpy_seed(worker_id, seed):
    torch_seed = torch.initial_seed() % 2**30
    return np.random.seed(torch_seed + worker_id + seed)


def set_numpy_seed_train(worker_id):
    return set_numpy_seed(worker_id, 0)


def set_numpy_seed_val(worker_id):
    return set_numpy_seed(worker_id, 1000)


def set_numpy_seed_test(worker_id):
    return set_numpy_seed(worker_id, 2000)


class DataModule(pl.LightningDataModule):

    def __init__(self, batch_size=128, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _get_dataloader(self, size: int, stage: str):
        return torch.utils.data.DataLoader(
            dataset.GenSVGDataset(size=size),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=globals()[f'set_numpy_seed_{stage}'],
            pin_memory=True  # good for CUDA
        )

    def train_dataloader(self):
        return self._get_dataloader(size=128 * 50000, stage='train')

    def val_dataloader(self):
        return self._get_dataloader(size=128, stage='val')

    def test_dataloader(self):
        return self._get_dataloader(size=128, stage='test')


class Model(pl.LightningModule):

    def __init__(self, learning_rate=3e-4, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = model.VisionTransformer(*args, **kwargs)

    def forward(self, x):
        return self.model(x)

    def loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def _evaluate(self, batch, stage):
        x, y = batch
        x_patches = self.model.img_to_patch(x)
        y_patches = self.model.img_to_patch(y)
        y_hat = self(x_patches)
        loss = self.loss(y_hat, y_patches)

        self.log(f'{stage}_loss', loss, prog_bar=True)
        if stage == 'train':
            return loss
        else:
            return x, y, y_hat

    def training_step(self, batch, batch_idx):
        loss = self._evaluate(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        return self._evaluate(batch, 'val')

    def validation_epoch_end(self, outputs):
        x, y, y_hat = outputs[0]

        # make all images with C=3 channels (B, C, H, W)
        y = y.repeat([1, 3, 1, 1])
        y_hat = self.model.patch_to_img(y_hat).repeat([1, 3, 1, 1])
        ims = torch.stack([x, y, y_hat]).transpose(1, 0).flatten(0, 1)
        # make a 3 x 5 grid of (x, y, y_hat)
        grid = torchvision.utils.make_grid(ims[:15], nrow=3)
        # log to tensorboard
        self.logger.experiment.add_image('images', grid, global_step=self.global_step)

    def test_step(self, batch, batch_idx):
        return self._evaluate(batch, 'test')

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(),
        #                               lr=self.hparams.learning_rate)
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.hparams.learning_rate,
                                    momentum=self.hparams.momentum)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=10)
        parser.add_argument('--momentum', type=float, default=.9)
        return parser


def main():
    pl.seed_everything(1, workers=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default='output', action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser = DataModule.add_argparse_args(parser)
    parser = Model.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if args.test:
        version = 'test'
        path = Path(args.output_path) / version
        if path.exists():
            shutil.rmtree(path)
    else:
        version = None

    data = DataModule.from_argparse_args(args)
    model = Model(**vars(args))

    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator='auto',
        default_root_dir=args.output_path,
        enable_checkpointing=False,
        max_epochs=1,
        val_check_interval=500,
        logger=pl.loggers.TensorBoardLogger(save_dir=args.output_path, name='', version=version)
    )
    trainer.fit(model, data)


if __name__ == '__main__':
    main()
