import torch 
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
from datetime import datetime
import time
import pickle

from params_and_dataloader import *
from audiocnn import CNN


# Some utility functions
#*************************************
def time_taken(elapsed):
    """To format time taken in hh:mm:ss. Use with time.monotic()"""
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

def save_obj(obj, name):
    with open(opt.outfolder + '/' + name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(opt.outfolder + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# Load model
#*************************************
cnn = CNN(opt.l1channels,opt.l2channels,opt.l3channels,opt.orientation,
          in_height,in_width,in_channels,n_classes,opt.kernelsize)

parameters={
    'in_height' : cnn.in_height, 
    'in_width' : cnn.in_width, 
    'in_channels' : cnn.in_channels, 
    'l1channels' : cnn.l1channels, 
    'l2channels' : cnn.l2channels,
    'l3channels' : cnn.l3channels, 
    'maxpool_kernel' : cnn.maxpool_kernel, 
    'conv_kernel' : cnn.conv_kernel, 
    'padding_size' : cnn.padding_size, 
    'downsampledheight' : cnn.downsampledheight, 
    'downsampledwidth' : cnn.downsampledwidth,
    'n_classes' : cnn.n_classes, 
    'orientation' : cnn.orientation,
    'kernelsize' : opt.kernelsize
}

if opt.cnn != '': # load checkpoint if needed
    cnn.load_state_dict(torch.load(opt.cnn))
print(cnn)
if opt.cuda:
    cnn.cuda()


# Loss and Optimizer
#*************************************
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=opt.lr)


# Train the model
#*************************************
def train(epoch):
    cnn.train() #put in training mode
    for i, (images, labels) in enumerate(train_loader):
        if opt.cuda:
            images, labels = images.cuda(), labels.cuda()
        images, labels = Variable(images), Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % opt.log_interval == 0:
        print ('{:%Y-%m-%d %H:%M:%S} Epoch [{}/{}], Iter [{}/{}] Loss: {:.4f}'.format( 
            datetime.now(), epoch+1, opt.num_epochs, i+1, len(train_dataset)//opt.batchsize, loss.data[0]))
        text_file2.write(str(loss.data[0]))
        text_file2.write('\n')


# Test the Model
#*************************************
def test():
    cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in test_loader:
        if opt.cuda:
            images, labels = images.cuda(), labels.cuda()
        images, labels = Variable(images, volatile=True), Variable(labels)
        
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1) #return index of max prob
        total += labels.data.shape[0]
        correct += (predicted == labels.data).sum()

    print('Test Accuracy of the model on %d test images: %.2f %%' % (len(test_dataset), 100 * correct / total))
    text_file3.write(str(100 * correct / total))
    text_file3.write('\n')


# Letsgo
#*************************************
text_file1 = open(opt.outfolder + '/' + opt.outfolder + '_parameters.txt', 'w') #save training data
text_file1.write(str(opt))
text_file1.write('\n')
text_file1.write(str(cnn))
text_file1.close()
save_obj(parameters,'params')

text_file2 = open(opt.outfolder + '/' + opt.outfolder + '_loss.txt', 'w') #save training data
text_file3 = open(opt.outfolder + '/' + opt.outfolder + '_accuracy.txt', 'w') #save training data
print('{:%Y-%m-%d %H:%M:%S} Starting training...'.format(datetime.now()))
start_time = time.monotonic()
for epoch in range(opt.num_epochs):
    train(epoch)
    test()
    # Save the Trained Model
    if epoch % opt.checkpoint_interval == 0:
        torch.save(cnn.state_dict(), '{}/{:%Y-%m-%d_%H-%M-%S}_cnn_epoch{}.pth'.format(opt.outfolder,datetime.now(),epoch))
elapsed_time = time.monotonic() - start_time
print('Training time taken:',time_taken(elapsed_time))
text_file2.close()
text_file3.close()

