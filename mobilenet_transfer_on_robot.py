#!/usr/bin/env python3
import time
import sys
sys.path.append("..")
import penguinPi as ppi
import pygame


import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def init_models():
    mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights)
    embedding_model = torch.nn.Sequential(*list(mobilenet.children()))[:-1]
    embedding_model.eval()
    classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features=576, out_features=1024, bias=True),
        torch.nn.Hardswish(),
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1024, out_features=3, bias=True)
    )
    classifier.load_state_dict('transfermodel.pt')
    classifier.eval()
    return embedding_model, classifier


def apply_model(embedding_model, classifier, image):
    with torch.no_grad():
        data = torch.tensor(image).unsqueeze(0)
        embeddings = embedding_model(data)
        embeddings2 = torch.nn.functional.adaptive_avg_pool2d(embeddings, (1, 1))
        embeddings3 = torch.flatten(embeddings2, 1)
        output = classifier(embeddings3)
        pred = torch.argmax(output, dim=1)

        if pred == 0:
            return 20, 40
        elif pred == 1:
            return 40, 20
        elif pred == 2:
            return 30, 30
        else: 
            assert False


if __name__=="__main__":
    #~~~~~~~~~~~~ SET UP Game ~~~~~~~~~~~~~~
    pygame.init()
    pygame.display.set_mode((300,300)) #size of pop-up window
    pygame.key.set_repeat(100) #holding a key sends continuous KEYDOWN events. Input argument is milli-seconds delay between events and controls the sensitivity
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
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        print("stop")                    
                        ppi.set_velocity(0,0)
                        raise KeyboardInterrupt
    #stops motors on shutdown
    except KeyboardInterrupt:
        ppi.set_velocity(0,0)
    except:
        raise