import imageio
import torch
from tqdm import tqdm
from animate import normalize_kp
from demo import load_checkpoints
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage import img_as_ubyte
from skimage.transform import resize
import cv2
import os
import argparse
import pyfakewebcam
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import time
from statistics import mean
import matplotlib.pyplot as plt
from pylive import live_plotter


ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--input_image", required=True,help="Path to image to animate")
ap.add_argument("-c", "--checkpoint", default='checkpoints/vox-adv-cpk.pth.tar', help="Path to checkpoint")
ap.add_argument("-v","--input_video", required=False, help="Path to video input")
ap.add_argument("-d","--debug", required=False, help="Show debug output", action='store_true')
ap.add_argument("--cpu", required=False, help="Using CPU computation", action='store_true')
ap.add_argument("--vc", required=False, help="Enable with virtualcam", action='store_true')
ap.add_argument("--csv", required=False, help="Export frame rate report in CVS file", action="store_true")

args = vars(ap.parse_args())

def main():
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    print(filename)
    process(filename)


def process(input):
    print("[INFO] loading source image and checkpoint...")
    source_path = input
    checkpoint_path = args['checkpoint']
    if args['input_video']:
        video_path = args['input_video']
    else:
        video_path = None
    source_image = imageio.imread(source_path)
    source_image = resize(source_image,(256,256))[..., :3]

    generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml', checkpoint_path=checkpoint_path)

    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if not os.path.exists('output'):
        os.mkdir('output')


    relative=True
    adapt_movement_scale=True
    if args['cpu']:
        cpu = True
    else:
        cpu = False

    if video_path:
        cap = cv2.VideoCapture(video_path) 
        print("[INFO] Loading video from the given path")
    else:
        cap = cv2.VideoCapture(0)
        print("[INFO] Initializing front camera...")
        # get vcap property 
        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print('resolution : {} x {}'.format(width,height))
        print('frame rate : {} \nframe count : {}'.format(fps, frame_count))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out1 = cv2.VideoWriter('output/test.avi', fourcc, 12, (256*3 , 256), True)

    cv2_source = cv2.cvtColor(source_image.astype('float32'),cv2.COLOR_BGR2RGB)
    cv2_source2 = (source_image*255).astype(np.uint8)

    if args['vc']:
        camera = pyfakewebcam.FakeWebcam('/dev/video7', 640, 360)
        camera._settings.fmt.pix.width = 640
        camera._settings.fmt.pix.height = 360

    img = np.zeros((360,640,3), dtype=np.uint8)
    yoff = round((360-256)/2)
    xoff = round((640-256)/2)
    img_im = img.copy()
    img_cv2_source = img.copy()
    img_im[:,:,2] = 255
    img_cv2_source[:,:,2] = 255
    with torch.no_grad() :
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        kp_source = kp_detector(source)
        count = 0
        fps = []
        if args['csv']:
            line1 = []
            size = 10
            x_vec = np.linspace(0,1,size+1)[0:-1]
            y_vec = np.random.randn(len(x_vec))
        while(True):
            start = time.time()
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect the faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            frame = cv2.flip(frame,1)
            if ret == True:
                
                if not video_path:
                    x = 143
                    y = 87
                    w = 322
                    h = 322 
                    frame = frame[y:y+h,x:x+w]
                frame1 = resize(frame,(256,256))[..., :3]
                
                if count == 0:
                    source_image1 = frame1
                    source1 = torch.tensor(source_image1[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                    kp_driving_initial = kp_detector(source1)
                
                frame_test = torch.tensor(frame1[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

                driving_frame = frame_test
                if not cpu:
                    driving_frame = driving_frame.cuda()
                kp_driving = kp_detector(driving_frame)
                kp_norm = normalize_kp(kp_source=kp_source,
                                    kp_driving=kp_driving,
                                    kp_driving_initial=kp_driving_initial, 
                                    use_relative_movement=relative,
                                    use_relative_jacobian=relative, 
                                    adapt_movement_scale=adapt_movement_scale)
                out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
                im = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
                #im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
                #cv2_source = cv2.cvtColor(cv2_source,cv2.COLOR_RGB2BGR)
                im = (np.array(im)*255).astype(np.uint8)
                #cv2_source = (np.array(cv2_source)*255).astype(np.uint8)
                img_im[yoff:yoff+256, xoff:xoff+256] = im
                img_cv2_source[yoff:yoff+256, xoff:xoff+256] = cv2_source2
                #print(faces)
                #print(type(im))
                if args['debug']:
                    #print("[DEBUG] FPS : ",1.0 / (time.time()-start))
                    fps.append(1.0 / (time.time()-start))
                    if args['cpu']:
                        print("[DEBUG] Avg. of FPS using CPU : ",mean(fps))
                    else:
                        print("[DEBUG] Avg. of FPS using GPU : ",mean(fps))

                if args['csv']:
                    y_vec[-1] = mean(fps)
                    line1 = live_plotter(x_vec,y_vec,line1)
                    y_vec = np.append(y_vec[1:],0.0)

                if args['vc']:
                    if np.array(faces).any():
                        #joinedFrame = np.concatenate((cv2_source,im,frame1),axis=1)
                        camera.schedule_frame(img_im)
                    else:
                        #joinedFrame = np.concatenate((cv2_source,cv2_source,frame1),axis=1)
                        camera.schedule_frame(img_cv2_source)
                    #cv2.imshow('Test',joinedFrame)
                    #out1.write(img_as_ubyte(np.array(im)))
                count += 1
            else:
                break

            
        cap.release()
        out1.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()