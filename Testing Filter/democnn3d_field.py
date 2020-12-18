import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

import time
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import glob
import cv2
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(f'Using {device}')
#input shape: nx1050x1680x3 or 1
#output shape: nx2 (x,y)


class ArmCameraDataset(Dataset):

    def __init__(self, arm_dir, overhead_dir):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass




def calculate_loss(network,loss_fn,data_generator):
    #calculate the average loss over the data
    network.eval()
    total_loss = 0
    n_samples = 0
    with torch.no_grad():
        for x_batch, y_batch in data_generator:
            x,y = x_batch.to(device), y_batch.to(device)
            output = network(x)
            loss = loss_fn(output,y)
            total_loss += (loss)*len(y_batch)
            n_samples += len(y_batch)
    network.train()
    return total_loss/n_samples

def log_statistics(network, loss_fn, trainloader, validationloader):
    # convenience function to calculate all the stats
    train_loss = calculate_loss(network, loss_fn, trainloader)
    val_loss = calculate_loss(network, loss_fn, validationloader)
    # train_err = 1-calculate_accuracy(network,trainloader)
    # val_err = 1-calculate_accuracy(network,validationloader)
    print(f"training loss: {train_loss}")
    # print(f"training accuracy: {1-train_err}")
    print(f"validation loss: {val_loss}")
    return train_loss, val_loss

class MnistNetwork(nn.Module):
    '''
    The core network used in most of these experiments
    '''
    def __init__(self):
        super().__init__()
        #2d convolution layer using (3x3) filter size, with 32 channels, and a ReLU activation
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(11,6)) # 1 input channel because grayscale
        #2d MaxPool layer with a (2x2) downsampling factor
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))
        #2d convolution layer using (3x3) filter size, with 32 channels, and a ReLU activation
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(6,4))
        self.conv2_bn = nn.BatchNorm2d(64)
        #2d MaxPool layer with a (2x2) downsampling factor
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3,3))
        #dense layer with 128-dimensional output and ReLU activation
        self.dense1 = nn.Linear(30*19*64,48*64) #based on output shapes below--is there some less hard-coded way to calculate this?
        # self.dense1_bn = nn.BatchNorm1d(128)
        #10d output and softmax activation, which maps to a distribution over 10 MNIST classes
        # self.dense2 = nn.Linear(128,2)
    def forward(self, x):
        x = x.view(-1,3,200,125) # reshaping
        x = self.maxpool1(F.leaky_relu(self.conv1(x))) # output shape is (1,32,1040/90/190,1040/90/120)--pool-->(1,32,260/45/95,260/45/60)
        x = self.maxpool2(F.leaky_relu(self.conv2_bn(self.conv2(x)))) # output shape is (1,32,252/40/90,252/40/57)--pool-->(1,32,30,19)
        x = F.leaky_relu(self.dense1(x.view(-1,30*19*64))) # flatten input to dense1, then relu output, output shape is (1,128)
        # x = F.softmax(self.dense2(x), dim=1) # softmaxing over [1,10] so dim=1 b/c we want the 10 outputs to sum to 1<<---REMOVE! softmax is in loss function
        # x = self.dense2(x)
        return x
    def predict(self, x):
        return F.softmax(self.forward(x),dim=1)
def plot_prediction(network,i,test=True):
    with torch.no_grad():
        overhead_path = 'overhead_pics/*'
        overhead_files = glob.glob(overhead_path)
        overhead_files.sort()
        overhead_img = cv2.imread(overhead_files[i], cv2.IMREAD_GRAYSCALE)
        if test:
            test_img,loc = test_data[0]
            pred = network(test_img[i].to(device)).cpu().numpy()[0]
        else:
            test_img,loc = train_data[0]
            pred = network(test_img[i].to(device)).cpu().numpy()[0]
        pred[0] *= 640
        pred[1] *= 480
        print((loc[i][0]*640, loc[i][1]*480))
        print(pred)
        fig,ax = plt.subplots(1)
        ax.imshow(overhead_img)
        circle = patches.Circle(pred, radius=5, edgecolor='r', facecolor='r')
        ax.add_patch(circle)
        plt.show()

def get_loc_field(loc):
    field = np.zeros((48,64))
    field[int(loc[1]//10),int(loc[0]//10)] = 1
    return field
if __name__ == '__main__':
    cnn = MnistNetwork()
    cnn.to(device)
    batchsize = 32
    rawimages = np.load('armpicscolor.npy')
    images = torch.split(torch.from_numpy(rawimages).float(), batchsize)
    rawlocations = np.load('locs.npy')
    rawlocfields = np.array([np.argmax(get_loc_field(loc).flatten()) for loc in rawlocations])
    
    # rawlocations[:,0] /= 640 #scale x to 0,1
    # rawlocations[:,1] /= 480 #scale y to 0,1
    locations = torch.split((torch.from_numpy(rawlocfields)).long(), batchsize)

    p = transforms.Compose([transforms.Scale(64,64)])

    
    test_data = list(zip(images[:2],locations[:2]))
    train_data = list(zip(images[2:], locations[2:])) # just split the first batch for test because the data's random anyway
    # train using SGD with momentum = 0.99, step size = 0.001, minibatch size=32, for 5 epochs
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 100
    train_losses = []
    val_losses = []
    training_time = 0
    # log statistics
    # train_loss, train_err, val_err = log_statistics(nn, loss_fn, trainloader, validationloader)
    # train_losses.append(train_loss)
    # train_errors.append(train_err)
    # val_errors.append(val_err)
    for epoch in range(epochs):
        start_time = time.time()
        for x_batch,y_batch in train_data:
            # print("batch")
            x,y = x_batch.to(device),y_batch.to(device)
            optimizer.zero_grad()
            output = cnn(x)
        
            loss = loss_fn(output, y)
        
            loss.backward()
            optimizer.step()
        epoch_time = time.time()-start_time
        training_time += epoch_time
        print(f"Epoch time: {epoch_time} s")
        # log statistics
        train_loss, val_loss = log_statistics(cnn, loss_fn, train_data, test_data)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    plot_prediction(cnn,0)
    import pdb; pdb.set_trace()
    

