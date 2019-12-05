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
			print(len(files), len(files_))
			if ((len(files)) < len(files_)):
				for filename in files_ :
					ext = os.path.splitext(filename)[-1]
					if ext == '.jpg':
						relF = random.uniform(fMin, fMax)
						relF = 0.1
						image = Image.open(os.path.join(ORIGINAL_DATA_DIR, filename)).resize((math.ceil(image_shape[0]), math.ceil(image_shape[1])))
						image = np.array(image)
						f = relF * math.sqrt((image_shape[0])**2+(image_shape[1])**2)
						scale = math.sqrt((image_shape[0])**2+(image_shape[1])**2)/f
						distortedImage = np.zeros(image_shape).astype(np.uint8)

						print(distortedImage.shape, image.shape)

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

learning_rate = 1e-4
training_epochs = 500
display_step = 20
batch_size = 64

image_shape = [360, 360, 3]

fMin = 1.25
fMax = 4

torch.set_default_dtype(torch.float32)
    
print(torch.version)
print(torch.version.cuda)

use_cuda = torch.cuda.is_available()
print(use_cuda)

get_dataset(image_shape, fMin, fMax)