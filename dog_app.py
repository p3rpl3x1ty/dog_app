#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFile
import cv2 

# importing Pytorch model libraries
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import requests

from io import BytesIO

classes = ['Affenpinscher', 'Afghan hound', 'Airedale terrier', 'Akita',
           'Alaskan malamute', 'American eskimo dog', 'American foxhound', 
           'American staffordshire terrier', 'American water spaniel', 
           'Anatolian shepherd dog', 'Australian cattle dog', 'Australian shepherd', 
	   'Australian terrier', 'Basenji', 'Basset hound', 'Beagle', 'Bearded collie', 
	   'Beauceron', 'Bedlington terrier', 'Belgian malinois', 'Belgian sheepdog', 
	   'Belgian tervuren', 'Bernese mountain dog', 'Bichon frise', 
	   'Black and tan coonhound', 'Black russian terrier', 'Bloodhound', 
	   'Bluetick coonhound', 'Border collie', 'Border terrier', 'Borzoi', 
	   'Boston terrier', 'Bouvier des flandres', 'Boxer', 'Boykin spaniel', 
	   'Briard', 'Brittany', 'Brussels griffon', 'Bull terrier', 'Bulldog', 
	   'Bullmastiff', 'Cairn terrier', 'Canaan dog', 'Cane corso', 
	   'Cardigan welsh corgi', 'Cavalier king charles spaniel', 
	   'Chesapeake bay retriever', 'Chihuahua', 'Chinese crested', 
	   'Chinese shar-pei', 'Chow chow', 'Clumber spaniel', 'Cocker spaniel', 
	   'Collie', 'Curly-coated retriever', 'Dachshund', 'Dalmatian', 
	   'Dandie dinmont terrier', 'Doberman pinscher', 'Dogue de bordeaux', 
	   'English cocker spaniel', 'English setter', 'English springer spaniel', 
	   'English toy spaniel', 'Entlebucher mountain dog', 'Field spaniel', 
	   'Finnish spitz', 'Flat-coated retriever', 'French bulldog', 'German pinscher', 
	   'German shepherd dog', 'German shorthaired pointer', 'German wirehaired pointer', 
	   'Giant schnauzer', 'Glen of imaal terrier', 'Golden retriever', 'Gordon setter', 
	   'Great dane', 'Great pyrenees', 'Greater swiss mountain dog', 'Greyhound', 
	   'Havanese', 'Ibizan hound', 'Icelandic sheepdog', 'Irish red and white setter', 
	   'Irish setter', 'Irish terrier', 'Irish water spaniel', 'Irish wolfhound', 
	   'Italian greyhound', 'Japanese chin', 'Keeshond', 'Kerry blue terrier', 'Komondor', 
	   'Kuvasz', 'Labrador retriever', 'Lakeland terrier', 'Leonberger', 'Lhasa apso', 
	   'Lowchen', 'Maltese', 'Manchester terrier', 'Mastiff', 'Miniature schnauzer', 
	   'Neapolitan mastiff', 'Newfoundland', 'Norfolk terrier', 'Norwegian buhund', 
	   'Norwegian elkhound', 'Norwegian lundehund', 'Norwich terrier', 
	   'Nova scotia duck tolling retriever', 'Old english sheepdog', 'Otterhound', 
	   'Papillon', 'Parson russell terrier', 'Pekingese', 'Pembroke welsh corgi', 
	   'Petit basset griffon vendeen', 'Pharaoh hound', 'Plott', 'Pointer', 'Pomeranian', 
	   'Poodle', 'Portuguese water dog', 'Saint bernard', 'Silky terrier', 'Smooth fox terrier', 
	   'Tibetan mastiff', 'Welsh springer spaniel', 'Wirehaired pointing griffon', 'Xoloitzcuintli', 'Yorkshire terrier']


def url_to_img(url):
    """
    Fetch an image from url and convert it into a Pillow Image object
    """
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    transform = transforms.Compose([transforms.Resize(255),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    img = transform(img)[:3,:,:].unsqueeze(0)
    
    return img

class PythonPredictor:
    def __init__(self):
        #Loading vgg11 into the variable model_transfer
        self.model = models.vgg11(pretrained=False)

        #Freezing the parameters
        for param in self.model.features.parameters():
            param.requires_grad = False
	   
        #Changing the classifier layer
        self.model.classifier[6] = nn.Linear(4096,133,bias=True)
	# Link to model_transfer.pt : https://drive.google.com/file/d/1-9XJwRBquErIzS94RmeSUp2LbYSvpzRL/view?usp=sharing
        self.model.load_state_dict(torch.load('model_transfer.pt', map_location=torch.device('cpu')))
        self.model.eval()

    def predict(self, payload):
        """ Run a model based on url input. """

        # Inference
        img = url_to_img(link)
        results = self.model(img)
        idx = torch.argmax(results)
        return classes[idx]
        
if __name__ == '__main__':
    p = PythonPredictor()
    link = 'https://images.pexels.com/photos/7210704/pexels-photo-7210704.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940'
    print(p.predict(link))
