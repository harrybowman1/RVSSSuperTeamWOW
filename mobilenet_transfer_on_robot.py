#!/usr/bin/env python3
import time
import sys
sys.path.append("..")
import penguinPi as ppi
from PIL import Image
# import pygame
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def init_models():
    mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    embedding_model = torch.nn.Sequential(*list(mobilenet.children())[:-1])
    embedding_model.eval()

    classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features=576, out_features=256, bias=True),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(in_features=256, out_features=128, bias=True),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(in_features=128, out_features=3, bias=True),
    )
    classifier.load_state_dict(torch.load('class_params_e19.pt'))
    classifier.eval()
    return embedding_model, classifier

DOWNSIZE = (32, 32)
HEIGHT = 240
WIDTH = 320
CROP_FRAC = 5
preprocess = transforms.Compose([
    transforms.Lambda(lambda x: transforms.functional.crop(x, int(HEIGHT/CROP_FRAC), 0, int(HEIGHT-HEIGHT/CROP_FRAC), WIDTH)),
    transforms.Resize(DOWNSIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def apply_model(embedding_model, classifier, image):
    with torch.no_grad():
        data = Image.fromarray(image*255)
        data_pp = preprocess(data)
        embeddings = embedding_model(data_pp.unsqueeze(0))
        embeddings2 = torch.nn.functional.adaptive_avg_pool2d(embeddings, (1, 1))
        embeddings3 = torch.flatten(embeddings2, 1)
        output = classifier(embeddings3)
        pred = torch.argmax(output, dim=1)

        if pred == 0:
            return 40, 20
        elif pred == 1:
            return 20, 40
        elif pred == 2:
            return 30, 30
        else: 
            assert False


if __name__=="__main__":
    #~~~~~~~~~~~~ SET UP Game ~~~~~~~~~~~~~~
    #pygame.init()
    #pygame.display.set_mode((300,300)) #size of pop-up window
    #pygame.key.set_repeat(100) #holding a key sends continuous KEYDOWN events. Input argument is milli-seconds delay between events and controls the sensitivity
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # stop the robot 
    ppi.set_velocity(0,0)

    # init
    print("initialise camera")
    camera = ppi.VideoStreamWidget('http://localhost:8080/camera/get')
    time.sleep(2)

    em, cm = init_models()

    print("controller start")
    try:
        # MAIN LOOP
        while True:
            image = camera.frame
            outs = apply_model(em, cm, image)
            ppi.set_velocity(outs[0],outs[1])
            # SPACE for shutdown 
            #for event in pygame.event.get():
            #    if event.type == pygame.KEYDOWN:
            #        if event.key == pygame.K_SPACE:
            #            print("stop")                    
            #            ppi.set_velocity(0,0)
            #            raise KeyboardInterrupt
    #stops motors on shutdown
    except KeyboardInterrupt:
        ppi.set_velocity(0,0)
    except:
        raise