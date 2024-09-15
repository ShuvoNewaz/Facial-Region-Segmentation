import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
import os
from src.pspnet import *
import torch
from src.data_transforms import get_transforms


cwd = os.getcwd()
live_image_dir = os.path.join(cwd, 'live_images_isolated')
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

extracting_face = False
extracting_part = False
text = 'Showing entire image.\nPress "f" to show face.'
label_translator = {1: 'skin', 2: 'eybrows', 3: 'eyes', 4: 'nose', 5: 'lips', 9: 'ears'}

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


def PartExtractor(face, label): # face is RGB
    face_gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    face_gray = torch.as_tensor(face_gray)
    face_gray = torch.unsqueeze(face_gray, 0)

    # transform

    face = torch.as_tensor(face)
    face = torch.permute(face, (2, 0, 1))
    face = transform(face)
    face_gray = transform(face_gray)
    face = face.numpy()

    face_gray = face_gray.to(device)
    seg, yhat, main, aux = face_extractor_model(face_gray, torch.zeros(face_gray.shape).to(device))
    seg = seg.cpu()
    main = main.cpu()
    aux = aux.cpu()
    face_gray = face_gray.cpu()
    yhat = yhat.cpu()

    # Extract Skin

    face_gray = torch.squeeze(face_gray, 0)
    yhat = torch.squeeze(yhat, 0)
    H, W = face_gray.shape
    part_image = np.zeros(face_gray.shape)
    part_ind_x, part_ind_y = np.where(yhat == label)

    if len(part_ind_x) == 0:
        part_ind_x = np.concatenate(((part_ind_x), np.array([0])))
        part_ind_y = np.concatenate(((part_ind_y), np.array([0])))

    part_xmin = np.min(part_ind_x, initial=H)
    part_xmax = np.max(part_ind_x, initial=0)
    part_ymin = np.min(part_ind_y, initial=W)
    part_ymax = np.max(part_ind_y, initial=0)

    if part_xmin > part_xmax:
        part_ymin = 0
        part_xmax = H
    if part_ymin > part_ymax:
        part_ymin = 0
        part_ymax = W

    w, h = part_ymax - part_ymin, part_xmax - part_xmin
    part_image[part_ind_x, part_ind_y] = face_gray[part_ind_x, part_ind_y]

    # Swap axes for proper broadcasting

    part_image_color = np.zeros((3, H, W))
    part_image_color = face * (part_image != 0)
    part_image_color = np.swapaxes(part_image_color, 0, 1)
    part_image_color = np.swapaxes(part_image_color, 2, 1)

    return part_image_color

vid = cv2.VideoCapture(0)  
while(True): 
    ret, frame = vid.read()
    correction_patch = frame[50:80, 280:320, :]
    cv2.imwrite(image_dir, frame)

    # Extract face

    face_objs = DeepFace.extract_faces(img_path = image_dir, 
                                                # target_size = (240, 240),
                                                detector_backend = backends[0],
                                                enforce_detection=False
                                                )
    facial_area = face_objs[0]['facial_area']
    x = facial_area['x']
    y = facial_area['y']
    w = facial_area['w']
    h = facial_area['h']
    frame = frame[y:y+h, x:x+w, :]

    fig = plt.figure(figsize=(9, 9))

    # Show facial region

    face = plt.imread(image_dir) # To show whole image
    keypress = cv2.waitKey(100)
    if keypress == ord('a'):
        text = 'Showing entire image.\nPress "f" to show face.'
        extracting_face = False
        extracting_part = False
    elif keypress == ord('f') or extracting_face or extracting_part:
        extracting_face = True
        if keypress == ord('f'):
            text = 'Showing only face.\nPress label number to show part.'
            extracting_part = False
        face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if keypress in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('9')]:
            label = int(chr(keypress))
            text = f'Showing only {label_translator[label]}'
            extracting_part = True
        if extracting_part:
            face = PartExtractor(face, label)
            extracting_face = False

    plt.imshow(face)
    plt.axis('off')
    
    plt.savefig(seg_dir, bbox_inches='tight')
    show_image = cv2.imread(seg_dir)
    text_offset = 30
    for i, line in enumerate(text.split('\n')):
        show_image = cv2.putText(show_image, line, org=(20, 50+i*text_offset), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1, color=(0, 0, 255), thickness=2)

    cv2.namedWindow('Image', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Image', show_image)

    if keypress == ord('q'): 
        break

vid.release()
cv2.destroyAllWindows() 