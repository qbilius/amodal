import torch
from torch import nn


class AttentionBlock(nn.Module):
    def __init__(self,
                 embed_dim,  # =3 * 3 * 4,
                 hidden_dim,  # =3 * 3 * 4 * 4,
                 num_heads,  # =4,
                 dropout,  # =.2
                 ):
        """Basic attention block

        Args:
            embed_dim (int, optional): Dimensionality of input and attention feature vectors. Defaults to 3*3*4.
            hidden_dim (int, optional): Dimensionality of hidden layer in feed-forward network (usually 2-4x larger than embed_dim). Defaults to 3*3*4*4.
            num_heads (int, optional): Number of heads to use in the Multi-Head Attention block. Defaults to 4.
            dropout (float, optional): Amount of dropout to apply in the feed-forward network. Defaults to .2.
        """

        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # breakpoint()
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        # embed_dim=3 * 3 * 3 * 2,
        # hidden_dim=3 * 3 * 4 * 4,
        image_size=32,
        num_channels=3,
        num_heads=4,
        num_layers=2,
        patch_size=4,
        # num_patches=...,
        dropout=.2,
        *args,
        **kwargs
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        inp_dim = num_channels * (patch_size**2)
        embed_dim = inp_dim * 2
        num_patches = (image_size // patch_size) ** 2

        self.embedding = nn.Linear(inp_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Sequential(
            *(AttentionBlock(
                embed_dim,
                embed_dim * 2,
                num_heads,
                dropout=dropout) for _ in range(num_layers))
        )
        self.prediction = nn.Linear(embed_dim, self.patch_size ** 2)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        x = self.prediction(x)
        return x

    def img_to_patch(self, x: torch.Tensor, flatten_channels=True) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): Tensor representing the image of shape [B, C, H, W]
            flatten_channels (bool, optional): If True, the patches will be returned in a flattened format as a feature vector instead of a image grid. Defaults to True.

        Returns:
            torch.Tensor: Patches of shape [B, H'*W', C*p_H*p_W]
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
        x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
        if flatten_channels:
            x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
        return x

    def patch_to_img(self, patches: torch.Tensor) -> torch.Tensor:
        """Converts patches back to an image

        Args:
            patches (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        batch_size = patches.shape[0]
        num_h = self.image_size // self.patch_size
        num_w = self.image_size // self.patch_size
        x = patches.reshape(batch_size, num_h, num_w, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.flatten(2, 3)
        x = x.flatten(3, 4)
        return x
