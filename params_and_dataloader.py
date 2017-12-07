import argparse
import os
import torch
import torchvision.transforms as transforms
import imgfoldergreyscale as iloader

# Hyper Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--trainfolder', required=True, help='path to training dataset')
parser.add_argument('--testfolder', required=True, help='path to dataset')
parser.add_argument('--outfolder', type=str, default=None, help='path to output (to save models etc.)')
parser.add_argument('--num_epochs', type=int, default=31, help='number of epochs to train for')
parser.add_argument('--batchsize', type=int, default=20, help='input batch size')
parser.add_argument('--orientation', type=str, choices=['2D','freq','time'], help='convolution over 1D (channels= freq or time bin) or 2D (channels=colour channel)', default='2D')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Adam optimizer, default=0.001')
parser.add_argument('--log_interval', type=int, default=1, help='log results every n steps')
parser.add_argument('--checkpoint_interval', type=int, default=10, help='log results every n epochs ')
parser.add_argument('--cnn', default='', help="path to cnn (to load existing model and continue training)")
parser.add_argument('--l1channels', type=int, help='Number of channels in the first convolutional layer', default=24) #default for testing
parser.add_argument('--l2channels', type=int, help='Number of channels in the second convolutional layer', default=48) #default for testing
parser.add_argument('--l3channels', type=int, help='Number of channels in the second convolutional layer', default=96) #default for testing
parser.add_argument('--kernelsize', type=int, help='Size of CNN kernel (filter). For 1D, dimension of kernel = [1,kernelsize], for 2D = [kernelsize,kernelsize]', default=5) #default for testing
parser.add_argument('--no_cuda', action='store_true', default=False, help='disable CUDA training')

opt = parser.parse_args()
print(opt)

if opt.outfolder is None:
    opt.outfolder = 'output'
os.system('mkdir {0}'.format(opt.outfolder))

opt.cuda = not opt.no_cuda and torch.cuda.is_available()
if opt.cuda:
    print('using CUDA backend...')
    
if opt.orientation == '2D':
    convert = transforms.Compose([ 
                    transforms.ToTensor()])
elif opt.orientation == 'freq':
    convert = transforms.Compose([ 
                    transforms.ToTensor(),
                    lambda x: x.permute(1,0,2).contiguous()])
elif opt.orientation == 'time':
    convert = transforms.Compose([ 
                    transforms.ToTensor(),
                    lambda x: x.permute(2,0,1).contiguous()])
    
# Prepare Dataset
train_dataset = iloader.ImageFolder(root=opt.trainfolder,
                              transform=convert)
                              
test_dataset = iloader.ImageFolder(root=opt.testfolder,
                              transform=convert)

# now grab the shape of data to correctly create the network later
in_height = train_dataset[0][0].shape[1] 
in_width = train_dataset[0][0].shape[2]
in_channels = train_dataset[0][0].shape[0]
n_classes = len(train_dataset.classes)

print("Reshaping with height = " + str(in_height) + ", width = " + str(in_width) + ", and channels = " + str(in_channels))

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=opt.batchsize, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=opt.batchsize, 
                                          shuffle=True)

print("Data successfully loaded...")