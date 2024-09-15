import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
import os
from src.pspnet import *
import torch
from src.data_transforms import get_transforms


cwd = os.getcwd()
live_image_dir = os.path.join(cwd, 'live_images_full')
image_dir = os.path.join(live_image_dir, 'frame.jpg')
seg_dir = os.path.join(live_image_dir, 'seg.jpg')

os.makedirs(live_image_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up the model

face_extractor_checkpoint_dir = os.path.join(cwd, 'models', 'face_weights.pt')
face_extractor_checkpoint = torch.load(face_extractor_checkpoint_dir)
face_extractor_model, face_extractor_optimizer = psp_model_optimizer(layers=50, num_classes=11)
face_extractor_model.eval()
face_extractor_model.load_state_dict(face_extractor_checkpoint['model_state_dict'])
face_extractor_model = face_extractor_model.to(device)
inp_size = [240, 240]
transform = get_transforms(inp_size=inp_size)


backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'fastmtcnn',
]

def segmenter(face): # face is RGB
    face_gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    face_gray = torch.as_tensor(face_gray)
    face_gray = torch.unsqueeze(face_gray, 0)

    # transform

    face_gray = transform(face_gray)
    face_gray = face_gray.to(device)
    seg, yhat, main, aux = face_extractor_model(face_gray, torch.zeros(face_gray.shape).to(device))

    # Empty GPU utilization

    seg = seg.cpu()
    main = main.cpu()
    aux = aux.cpu()
    face_gray = face_gray.cpu()
    yhat = yhat.cpu().squeeze(0)

    return yhat

def gridCount(n_total):
    if n_total == 0:
        nrow, ncol = 1, 1
    else:
        ncol = np.ceil(np.sqrt(n_total)).astype(int)
        nrow = np.ceil(n_total / ncol).astype(int)

    return nrow, ncol

vid = cv2.VideoCapture(0)
while(True): 
    ret, frame = vid.read()
    cv2.imwrite(image_dir, frame)

    # Extract face

    face_objs = DeepFace.extract_faces(img_path = image_dir, 
                                                # target_size = (240, 240),
                                                detector_backend = backends[4], # Use 0 for fast inference, 4 for accurate inference.
                                                enforce_detection=False
                                                )
    fig = plt.figure(figsize=(9, 9))
    num_faces = len(face_objs)
    nrow, ncol = gridCount(num_faces)
    if num_faces > 0:
        for i in range(len(face_objs)):
            facial_area = face_objs[i]['facial_area']
            x = facial_area['x']
            y = facial_area['y']
            w = facial_area['w']
            h = facial_area['h']
            face = frame[y:y+h, x:x+w, :]
        
            # Show facial region

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            regions = segmenter(face)
            plt.subplot(nrow, ncol, i + 1)
            plt.imshow(regions)
            plt.axis('off')
    else:
        plt.subplot(nrow, ncol, i + 1)
        plt.imshow(frame) # Show the whole image if face is not detected
        plt.axis('off')
    
    plt.savefig(seg_dir, bbox_inches='tight')
    show_image = cv2.imread(seg_dir)
    cv2.namedWindow('Image', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Image', show_image)
    keypress = cv2.waitKey(100)
    if keypress == ord('q'): 
        break

vid.release()
cv2.destroyAllWindows() 