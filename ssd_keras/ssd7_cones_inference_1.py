# Read a video and try to run ssd7 inference on it
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd7 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
set_session(tf.Session(config=config))

import cv2
import numpy as np
from matplotlib import pyplot as plt 

def main():
    img_height = 300
    img_width = 480
    n_classes = 5
    ### Load model
    LOAD_MODEL = True

    if LOAD_MODEL:
        # TODO: Set the path to the `.h5` file of the model to be loaded.
        model_path = '../ConeData/SavedModels/training3/(ssd7_epoch-10_loss-0.3291_val_loss-0.2664.h5'

        # We need to create an SSDLoss object in order to pass that to the model loader.
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

        K.clear_session() # Clear previous models from memory.

        model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                    'DecodeDetections': DecodeDetections,
                                                    'compute_loss': ssd_loss.compute_loss}) 

    ### Read video
    # cap = cv2.VideoCapture('test_videos/Building Self Driving Car - Local Dataset - Day.mp4')
    #cap = cv2.VideoCapture('test_videos/original.m4v')
    cap = cv2.VideoCapture('test_videos/20180619_175221224.mp4')
    width = int(cap.get(3))
    height = int(cap.get(4))
    property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cv2.VideoCapture.get(cap, property_id))
    count = 0

    detect = True

    while (count<total_frames):
        #print(str(j)+'/'+str(total_frames))
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if ret == True:
        #     cv2.imshow('original frame', frame)
        #     cv2.waitKey(10)

        if detect == True:
            frame = frame[...,::-1]
            frame_resized = cv2.resize(frame, (480, 300)) 
            frame_tensor = np.expand_dims(frame_resized, axis=0)
            ### Make predictions
            y_pred = model.predict(frame_tensor)
            y_pred_decoded = decode_detections(y_pred,
                                        confidence_thresh=0.75,
                                        iou_threshold=0.45,
                                        top_k=200,
                                        normalize_coords=True,
                                        img_height=img_height,
                                        img_width=img_width)
            
            #plt.figure(figsize=(20,12))
            #plt.imshow(frame_resized)

            #current_axis = plt.gca()

            ### plot predictions
            colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist() # Set the colors for the bounding boxes
            #classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'light'] # Just so we can print class names onto the image instead of IDs
            classes = ['background', 'cone'] # Just so we can print class names onto the image instead of IDs

            # Draw the predicted boxes in blue
            for box in y_pred_decoded[0]:
                xmin = int(box[-4])
                ymin = int(box[-3])
                xmax = int(box[-2])
                ymax = int(box[-1])

                #convert to x,y,w,h format
                # x_bbox = int(xmin)
                # y_bbox = int(ymin)
                # w_bbox = abs(int(xmax - xmin))
                # h_bbox = abs(int(ymax - ymin))
                
                color = colors[int(box[0])]
                cv2.rectangle(frame_resized,(xmin,ymin), (xmax,ymax), (0,255,0), 5)

                label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
                # current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
                # current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
                cv2.putText(frame_resized, label, (int(xmin),int(ymin)-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            
            #plt.savefig('output_frames/video_frame'+str(j)+'.png')
            #plt.close('all')
            
            #if j % 10 == 0:
            #clear_output()
        cv2.imshow('ssd7_inference', frame_resized)
        cv2.waitKey(10)

        count = count +1 
            
        
        # Break the loop
        #else: 
            #break

        #out.release()
    cap.release()

if __name__ == '__main__':
    main()