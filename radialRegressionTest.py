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
DATA_DIR = os.path.join(os.getcwd(), 'distorted_0.1to1Test')
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
	iteration = 501
	center = [math.floor(x/2) for x in image_shape]
	for (path, dir, files) in os.walk(DATA_DIR): 
		for (path_, dir_, files_) in os.walk(ORIGINAL_DATA_DIR):
			random.shuffle(files_)
			if len(files) < 500:
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
						iteration = iteration - 1
					if iteartion < 1:
						break

def make_image_batch(fileNameBatch):
	images = np.array([cv2.imread(filename) for filename in fileNameBatch])
	images = np.rollaxis(images, 3, 1)
	images = images/255
	return images

image_shape = [224, 224, 3]

fMin = 0.1
fMax = 1
print("start getting dataset")
data, labels = get_dataset(image_shape, fMin, fMax)
print("end getting dataset")
c = list(zip(data, labels))
random.shuffle(c)
data, labels = zip(*c)

train_test_ratio = 7

# Testing


torch.set_default_dtype(torch.float32)
print("torch version", torch.version)
use_cuda = torch.cuda.is_available()
print("use_cuda", use_cuda)

def make_image_batch(fileNameBatch):
	images = np.array([cv2.imread(filename) for filename in fileNameBatch])
	images = np.rollaxis(images, 3, 1)
	images = images/255
	return images


def next_batch(num, data, labels, i, batch_size):
	return np.asarray(data[i*batch_size:i*batch_size+batch_size]), np.asarray(labels[i*batch_size:i*batch_size+batch_size])

###############
'''
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
Deep Residual Learning for Image Recognition.
In CVPR, 2016.
'''

import torch.utils.model_zoo as model_zoo

def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
		    identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class CifarResNet(nn.Module):

	def __init__(self, block, layers, num_classes=10, use_cuda=False):
		super(CifarResNet, self).__init__()
		self.inplanes = 16
		self.conv1 = conv3x3(3, 16)
		self.bn1 = nn.BatchNorm2d(16)
		self.relu = nn.ReLU(inplace=True)

		self.layer1 = self._make_layer(block, 16, layers[0])
		self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(64 * block.expansion, num_classes)
		if use_cuda:
			self.conv1.cuda()
			self.bn1.cuda()
			self.relu.cuda()
			self.layer1.cuda()
			self.layer2.cuda()
			self.layer3.cuda()
			self.avgpool.cuda()
			self.fc.cuda() 

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x

def resnet_model():
	model = CifarResNet(BasicBlock, [1, 1, 1], num_classes=10, use_cuda=use_cuda)
	return model



###############
print("Start learning")
lossFunction = torch.nn.MSELoss() # TODO : change loss to mse 
if use_cuda:
    lossFunction.cuda()
batch_size = 2
img_shape = (3, 224, 224)
# Initialize generator and discriminator
regression = resnet_model()
num_ftrs = regression.fc.in_features
regression.fc = nn.Sequential(
	nn.Linear(num_ftrs, 50),
	nn.Linear(50, 1)
)
if use_cuda:
	regression.fc.cuda()
	regression.cuda()
Tensor = torch.FloatTensor
if use_cuda:
	Tensor = torch.cuda.FloatTensor

regression.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'fMin'+str(fMin)+'fMax'+str(fMax)+'checkpoint.pt')))
regression.eval()
epochLoss = 0.0
test_losses = []
for i in range(0, len(data_test)//batch_size): 
	torch.cuda.empty_cache()  
	filename_batch, labels_batch = next_batch(batch_size, data_test, labels_test, i, batch_size)
	data_batch = make_image_batch(filename_batch)
	labels_batch = labels_batch.reshape(-1, 1)
	output = regression(Tensor(data_batch))
	loss = lossFunction(output, Tensor(labels_batch))
	epochLoss = epochLoss + loss.data.numpy()
	if i % (len(data_test)//10) == 0:
		print("test real data / predicted data ", labels_batch, output, filename_batch)
print("Test - epoch: %d, loss: %f" % (epoch, epochLoss/len(data_test)))
test_losses.append(epochLoss/len(data_test))

torch.save(regression.state_dict(), os.path.join(MODEL_DIR, 'fMin'+str(fMin)+'fMax'+str(fMax)+'model.pt'))
