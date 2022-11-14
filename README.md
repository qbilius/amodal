# Amodal completion with transformers   
 
## Description  

An experiment to see if we can train [ViT](https://arxiv.org/abs/2010.11929) to output amodally completed shapes. This ViT implementation is based on [PyTorch Lightning Tutorial 11](https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/11-vision-transformer.html)

## How to run   

First, install dependencies   

```bash
# clone project   
git clone https://github.com/qbilius/amodal

# install project   
cd amodal 
pip install -e .   
pip install -r requirements.txt
```   

Now, run training locally: `python amodal/train.py`
Observe results with tensorboard: `tensorboard --logdir=output`

## License

MIT