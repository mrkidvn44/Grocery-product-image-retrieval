from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import io, base64
from PIL import Image
import json
import torch
from torch import nn
import torchvision.transforms as T
import faiss
from torchvision import datasets
from pathlib import Path
import numpy


# Because pythonanywhere doesn't alloy multithread
torch.set_num_threads(1)


# Super parameter for model to import
k = 32
compression_factor = 0.5

# Define class for model
class DenseLayer(nn.Module):

    def __init__(self,in_channels):

        super().__init__()

        self.BN1 = nn.BatchNorm2d(num_features = in_channels)
        self.conv1 = nn.Conv2d( in_channels=in_channels ,
                                out_channels=4*k ,
                                kernel_size=1 ,
                                stride=1 ,
                                padding=0 ,
                                bias = False )

        self.BN2 = nn.BatchNorm2d(num_features = 4*k)
        self.conv2 = nn.Conv2d( in_channels=4*k ,
                                out_channels=k ,
                                kernel_size=3 ,
                                stride=1 ,
                                padding=1 ,
                                bias = False )

        self.relu = nn.ReLU()

    def forward(self,x):

        xin = x

        # BN -> relu -> conv(1x1)
        x = self.BN1(x)
        x = self.relu(x)
        x = self.conv1(x)

        # BN -> relu -> conv(3x3)
        x = self.BN2(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = torch.cat([xin,x],1)

        return x


class DenseBlock(nn.Module):
    def __init__(self,layer_num,in_channels):

        super().__init__()
        self.layer_num = layer_num
        self.deep_nn = nn.ModuleList()

        for num in range(self.layer_num):
            self.deep_nn.add_module(f"DenseLayer_{num}",DenseLayer(in_channels+k*num))


    def forward(self,x):


        for layer in self.deep_nn:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    def __init__(self,in_channels,compression_factor):

        super().__init__()
        self.BN = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels = in_channels , out_channels = int(in_channels*compression_factor) ,kernel_size = 1 ,stride = 1 ,padding = 0, bias=False )
        self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)

    def forward(self,x):
        """
        Args:
            x (tensor) : input tensor to be passed through the dense block

        Attributes:
            x (tensor) : output tensor
        """
        x = self.BN(x)
        x = self.conv1(x)
        x = self.avgpool(x)
        return x


class DenseNet(nn.Module):
    def __init__(self,densenet_variant,in_channels,num_classes=10):


        super().__init__()

        # 7x7 conv with s=2 and maxpool
        self.conv1 = nn.Conv2d(in_channels=in_channels ,out_channels=64 ,kernel_size=7 ,stride=2 ,padding=3 ,bias = False)
        self.BN1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


        # adding 3 DenseBlocks and 3 Transition Layers
        self.deep_nn = nn.Sequential()
        dense_block_inchannels = 64

        for num in range(len(densenet_variant))[:-1]:
          if num:
            self.deep_nn.add_module( f"DenseBlock_{num+1}" , DenseBlock( densenet_variant[num] , dense_block_inchannels ) )
            dense_block_inchannels  = int(dense_block_inchannels + k*densenet_variant[num])

            self.deep_nn.add_module( f"TransitionLayer_{num+1}" , TransitionLayer( dense_block_inchannels,compression_factor ) )
            dense_block_inchannels = int(dense_block_inchannels*compression_factor)

        # adding the 4th and final DenseBlock
        self.deep_nn.add_module( f"DenseBlock_{num+2}" , DenseBlock( densenet_variant[-1] , dense_block_inchannels ) )
        dense_block_inchannels  = int(dense_block_inchannels + k*densenet_variant[-1])

        self.BN2 = nn.BatchNorm2d(num_features=dense_block_inchannels)

        # Average Pool
        self.average_pool = nn.AdaptiveAvgPool2d(1)

        # fully connected layer
        self.fc1 = nn.Linear(dense_block_inchannels, num_classes)

    def forward(self,x):
        x = self.relu(self.BN1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.deep_nn(x)

        x = self.relu(self.BN2(x))
        x = self.average_pool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        # print(x.shape)
        return x
# Model class end here

# Create hook to get embedding of image
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


# Custom dataset function to store the path to image on server
class ImageFolderWithPaths(datasets.ImageFolder):

    def __getitem__(self, index):

        img, label = super(ImageFolderWithPaths, self).__getitem__(index)

        path = self.imgs[index][0]

        return (img, label ,path)


data_transform = T.Compose([
        T.Resize(size = (256, 256)),
        T.RandomHorizontalFlip(p = 0.5),
        T.ToTensor(),])



my_transforms = T.Compose([T.Resize((256,256)),
                    T.ToTensor(),])

to3channel = T.Lambda(lambda x: x[:3])

# Convert from byte to image.
def transform_image(image_bytes):
    image = Image.open(io.BytesIO(base64.decodebytes(bytes(image_bytes, "utf-8"))))
    image = my_transforms(image)
    # Slice off the alpha channel from image if needed
    if len(image.shape) > 2 and image.shape[0] == 4:
        image = to3channel(image)
    return image.unsqueeze(dim = 0)

# Old image converter
# # Assuming base64_str is the string value without 'data:image/jpeg;base64,'
# def convert_to_img(base64_str):
#     img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
#     return img

def imageList(paths, labels):
  myList = []

  for path, label in zip(paths, labels):
        tmp_dict = {
        "src": "https://nhatphong.pythonanywhere.com/static/",
        "width": 255,
        "height": 255,
        "thumbnailCaption": "Label"}
        path = path.split('/')
        tmp_dict['src'] = tmp_dict['src'] + path[-2] + '/' + path[-1]
        tmp_dict['thumbnailCaption'] = label
        myList.append(tmp_dict)
  return json.dumps(myList)



# Load the pretrained model
model = DenseNet([1,2,4,3],3, 48)
model.load_state_dict(torch.load('/home/Nhatphong/mysite/model.pt',map_location=torch.device('cpu')))
model.BN2.register_forward_hook(get_activation('BN2'))


data_path = Path('/home/Nhatphong/mysite/data/content/image_downsample')

# Load image into a dataset
dataset = ImageFolderWithPaths(root = data_path,
                               transform = data_transform,
                               target_transform= None)

classes = [key for key in list(dataset.class_to_idx.keys())]

# Load the pretrained faiss index
index = faiss.read_index('/home/Nhatphong/mysite/index.pt')


# App code start here
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# Listening to POST and GET method
@app.route('/', methods=['POST','GET'])
@cross_origin()
def hi():
    paths = []
    labels = []
    if request.method == 'POST':
        content = request.json
        img = transform_image(content['imgBase64'])
        output = model(img)
        test_fea = activation['BN2']
        f_dists, f_ids = index.search(test_fea.reshape(1, 49152).detach().cpu().numpy(), k=20)
        result_ids = f_ids[0]
        for i in range(20):
            paths.append(dataset[result_ids[i]][2])
            labels.append(classes[dataset[result_ids[i]][1]])
    return jsonify(
        imageList(paths, labels)
    )

