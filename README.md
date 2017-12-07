## audio_CNN

Generic convolutional neural network for audio classification with spectrograms
Developed in Pytorch

classification.py: main script to run classification
audiocnn.py: model architecture is kept here
params_and_dataloader.py: argparse variables initialized here. Contain the dataloader method which imports from imgfoldergreyscale.py. As name suggests this dataloader is modified from the folder method in the Pytorch repo to read in greyscale (single-channel) images.

Example use:  

**To continue training with a 2D CNN from an exiting checkpoint. Training images kept in /data/train and test images in /data/test**
```bash
python classification.py --trainfolder ./data/train --testfolder ./data/test --outfolder 2d --orientation 2D --l1channels 18 --l2channels 36 --l3channels 72 --num_epochs 31 --kernelsize 11 --cnn ./2d/2017-12-07_13-31-05_cnn_epoch20.pth
```
