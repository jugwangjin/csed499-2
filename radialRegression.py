import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import cv2
import math
import random
import re
from PIL import Image

MODEL_DIR = os.path.join(os.getcwd(), 'generated_model')
DATA_DIR = os.path.join(os.getcwd(), 'data')
ORIGINAL_DATA_DIR = os.path.join(os.getcwd(), 'original_data')

def get_dataset(image_shape, fMin, fMax):
	make_dataset(image_shape, fMin, fMax)
	data = []
	labels = []
	for (path, dir, files) in os.walk(DATA_DIR):
		random.shuffle(files)
		for filename in files:
			ext = os.path.splitext(filename)[-1]
			if ext == '.jpg':
				label = re.search(r'\d+', filename)[0]
				labels.append(int(label) / 1000000.0)
				data.append(path+'/'+filename)
	return data, labels

def make_dataset(image_shape, fMin, fMax):
	center = [math.floor(x/2) for x in image_shape]
	for (path, dir, files) in os.walk(DATA_DIR): 
		for (path_, dir_, files_) in os.walk(ORIGINAL_DATA_DIR):
			if len(files) < len(files_):
				for filename in files_ :
					ext = os.path.splitext(filename)[-1]
					if ext == '.jpg':
						relF = random.uniform(fMin, fMax)
						image = Image.open(os.path.join(ORIGINAL_DATA_DIR, filename)).resize((math.ceil(image_shape[0]), math.ceil(image_shape[1])))
						image = np.array(image)
						f = relF * math.sqrt((image_shape[0])**2+(image_shape[1])**2)
						scale = math.sqrt((image_shape[0])**2+(image_shape[1])**2)/f
						distortedImage = np.zeros(image_shape).astype(np.uint8)
						for i in range(0, image_shape[0]):
							for j in range(0, image_shape[1]):
								relI = (i-center[0]) * math.sqrt(relF)
								relJ = (j-center[1]) * math.sqrt(relF)
								theta = math.atan2(relI, relJ)
								r = math.sqrt((relI*relI)+(relJ*relJ))
								if r <= f:
									rPrime = math.asin(r/f) * f
									x = math.floor(image.shape[0]/2 + rPrime*math.sin(theta))
									y = math.floor(image.shape[1]/2 + rPrime*math.cos(theta))
									if(x >= 0 and x < image.shape[0] and y >= 0 and y < image.shape[1]):
										for k in range(0, 3):
											distortedImage[i][j][k] = image[x][y][k]
						im = Image.fromarray(distortedImage)
						im.save(os.path.join(DATA_DIR, 'relF'+str(math.floor(relF*1000000))+'distorted'+filename))

def make_image_batch(fileNameBatch):
	images = np.array([cv2.imread(filename) for filename in fileNameBatch])
	images = np.rollaxis(images, 3, 1)
	images = images/255
	return images

image_shape = [250, 250, 3]

fMin = 0.1
fMax = 1

data, labels = get_dataset(image_shape, fMin, fMax)

c = list(zip(data, labels))
random.shuffle(c)
data, labels = zip(*c)

train_test_ratio = 7

data_val = data[math.floor((len(data)/(train_test_ratio+1))*train_test_ratio):]
labels_val = labels[math.floor((len(data)/(train_test_ratio+1))*train_test_ratio):]


data_train = data[:math.floor((len(data)/(train_test_ratio+1))*train_test_ratio)]
labels_train = labels[:math.floor((len(data)/(train_test_ratio+1))*train_test_ratio)]

data_test = data_train[math.floor((len(data_train)/(train_test_ratio+1))*train_test_ratio):]
labels_test = labels_train[math.floor((len(data_train)/(train_test_ratio+1))*train_test_ratio):]

data_train = data_train[:math.floor((len(data_train)/(train_test_ratio+1))*train_test_ratio)]
labels_train = labels_train[:math.floor((len(data_train)/(train_test_ratio+1))*train_test_ratio)]


# learning



torch.set_default_dtype(torch.float32)
print("torch version", torch.version)


def make_image_batch(fileNameBatch):
    images = np.array([cv2.imread(filename) for filename in fileNameBatch])
    images = np.rollaxis(images, 3, 1)
    images = images/255
    return images

class FRegression(nn.Module):
    
    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(FRegression, self).__init__()
        #input : 3*480*480
        conv1_1 = nn.Conv2d(3, 16, 3, 1, 1) # 16 * 480 * 480
        conv1_2 = nn.Conv2d(16, 32, 3, 1, 1) # 96 * 160 * 160
        pool1 = nn.MaxPool2d(3) # 128 * 80 *80

        conv2_1 = nn.Conv2d(32, 64, 3, 1, 1) # 128 * 160 * 160
        conv2_2 = nn.Conv2d(64, 96, 5, 1, 2) # 160 * 80 * 80
        pool2 = nn.MaxPool2d(2) # 128 * 80 *80
        
        conv3_1 = nn.Conv2d(96, 128, 5, 1, 2) # 192 * 80 * 80
        pool3 = nn.MaxPool2d(2) # 192 * 40 *40
        
        conv4_1 = nn.Conv2d(128, 128, 5, 1, 2) # 256 * 40 * 40
        pool4 = nn.MaxPool2d(4) # 256 * 20 *20
        
        self.conv_module = nn.Sequential(
            conv1_1,
            conv1_2,
            nn.LeakyReLU(),
            pool1,
            conv2_1,
            conv2_2,
            nn.LeakyReLU(),
            pool2,
            conv3_1,
            nn.LeakyReLU(),
            pool3,
            conv4_1,
            nn.LeakyReLU(),
            pool4
        )
        fc1 = nn.Linear(128*10*10, 256)
        fc2 = nn.Linear(256, 64)
        fc3 = nn.Linear(64, 1)
        
        self.fc_module = nn.Sequential(
            fc1,
            nn.LeakyReLU(),
            fc2,
            nn.LeakyReLU(),
            fc3,
            nn.LeakyReLU()
        )

        # gpu로 할당

    def forward(self, x):
        out = self.conv_module(x) # @192*10*10
        dim = 1
        for d in out.size()[1:]: #192*10*10
            dim = dim * d
        out = out.view(-1, dim)
        out = self.fc_module(out)
        return out

def next_batch(num, data, labels, i, batch_size):
    return np.asarray(data[i*batch_size:i*batch_size+batch_size]), np.asarray(labels[i*batch_size:i*batch_size+batch_size])


lossFunction = torch.nn.MSELoss() # TODO : change loss to mse 
epochs = 500
lr = 0.0001
batch_size = 128
display_step = 50
img_shape = (3, 250, 250)
# Initialize generator and discriminator
regression = FRegression()
# Optimizers
optimizer = torch.optim.Adam(regression.parameters(), lr=lr, weight_decay=lr/1000)
Tensor = torch.FloatTensor


best_validation_loss = 0.0
train_losses = []
test_losses = []

p = 15
n = 0
for epoch in range(epochs):
    c = list(zip(data_train, labels_train))
    random.shuffle(c)
    data_train, labels_train = zip(*c)
    epochLoss = 0.0
    
    for i in range(0, (len(data_train)//batch_size)):  
        filename_batch, labels_batch = next_batch(batch_size, data_train, labels_train, i, batch_size)
        data_batch = make_image_batch(filename_batch)
        labels_batch = labels_batch.reshape(-1, 1)
        optimizer.zero_grad()
        output = regression(Tensor(data_batch))
        loss = lossFunction(output, Tensor(labels_batch))
        loss.backward()
        optimizer.step() 
        epochLoss = epochLoss + loss.data.numpy()
    
    if (epoch+1) % display_step == 0:
    	print("epoch: %d, loss: %f" % (epoch, epochLoss/len(data_train)))
    train_losses.append(epochLoss/len(data_train))
    
    valid_loss = 0.0
    valid_num = 0
    for j in range(0, (len(data_val)//batch_size)):  
        filename_batch, labels_batch = next_batch(batch_size, data_val, labels_val, j, batch_size)
        data_batch = make_image_batch(filename_batch)
        labels_batch = labels_batch.reshape(-1, 1)
        optimizer.zero_grad()
        output = regression(Tensor(data_batch))
        loss = lossFunction(output, Tensor(labels_batch))
        valid_loss = valid_loss + loss.data.numpy()
        valid_num = valid_num + 1

	if epoch == 0:
      best_validation_loss = valid_loss/valid_num
    elif valid_loss/valid_num < best_validation_loss:
      if n > p//2: # if print every update, it makes too many prints
        print(f'Saving model, with validation loss ({best_validation_loss:.6f} --> {valid_loss/valid_num:.6f}).')
      torch.save(regression.state_dict(), os.path.join(MODEL_DIR, 'fMin'+str(fMin)+'fMax'+str(fMax)+'checkpoint.pt'))
      n = 0
      best_validation_loss = valid_loss/valid_num
    else:
      n = n + 1
    if n > p:
      print(f'Stopping training on epoch {epoch:d}')
      break

model2.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'fMin'+str(fMin)+'fMax'+str(fMax)+'checkpoint.pt')))

epochLoss = 0.0
for i in range(0, len(data_test)//batch_size):   
    filename_batch, labels_batch = next_batch(batch_size, data_test, labels_test, i, batch_size)
    data_batch = make_image_batch(filename_batch)
    labels_batch = labels_batch.reshape(-1, 1)
    output = regression(Tensor(data_batch))
    loss = lossFunction(output, Tensor(labels_batch))
    epochLoss = epochLoss + loss.data.numpy()
    if i % (len(data_test)//2) == 0:
    	print("test real data / predicted data ", labels_batch, output)

print("Test - epoch: %d, loss: %f" % (epoch, epochLoss/len(data_test)))
test_losses.append(epochLoss/len(data_test))

torch.save(regression.state_dict(), os.path.join(MODEL_DIR, 'fMin'+str(fMin)+'fMax'+str(fMax)+'model.pt'))
print("train losses" ,train_losses)
print("test losses", test_losses)