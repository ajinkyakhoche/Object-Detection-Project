from datasets import imagenet

import cv2
import numpy as numpy
import tensorflow as tf
from utils import label_map_util
# setup path
import sys
sys.path.append('/content/models/research/slim')

def main():
    cap = cv2.VideoCapture('./test_videos/20180619_175221224.mp4')
    #cap = cv2.VideoCapture('./test_videos/Formula Student Spain 2015 Endurance- DHBW Engineering with the eSleek15.mp4')
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    count = 0

    gd = tf.GraphDef.FromString(open('./frozen_weights/resnetv2_imagenet_frozen_graph.pb', 'rb').read())
    inp, predictions = tf.import_graph_def(gd,  return_elements = ['input:0', 'MobilenetV2/Predictions/Reshape_1:0'])

    with tf.Session(graph=inp.graph):
        while count < frameCount:
            ret, image_np = cap.read()
            if ret == True:
                count = count + 1
                img = cv2.resize(image_np, (224,224))
                x = predictions.eval(feed_dict={inp: img.reshape(1, 224,224, 3)})

                label_map = imagenet.create_readable_names_for_imagenet_labels()  
                print("Top 1 Prediction: ", x.argmax(),label_map[x.argmax()], x.max())

if __name__ == '__main__':
    main()