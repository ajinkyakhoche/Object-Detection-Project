import cv2
import numpy as numpy
import tensorflow as tflow
from utils import label_map_util
#from ConeDetection import *
from cone_img_processing2 import *
import os


# Set threshold for detection of cone for object detector
threshold_cone = 0.5

#Set path to check point and label map
#PATH_TO_CKPT = './frozen_orange_net.pb'
PATH_TO_CKPT = './frozen_weights/frozen_cone_graph_modified.pb'
#PATH_TO_CKPT = './frozen_weights/mobilenet_v2_0.75_224_frozen.pb'  
PATH_TO_LABELS = './test_scripts/label_map.pbtxt'

#Define no, of classes
NUM_CLASSES = 1         #only one class, i.e. cone

## Load a (frozen) Tensorflow model into memory.
detection_graph = tflow.Graph()
with detection_graph.as_default():
  od_graph_def = tflow.GraphDef()
  with tflow.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tflow.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

gpu_options = tflow.GPUOptions(per_process_gpu_memory_fraction=0.4)
#config=tflow.ConfigProto(gpu_options=gpu_options)

def mainLoop():
    # Try the following videos: 
    # 20180619_175221224    # shade to brightness
    # 20180619_180755490    # towards sun
    # 20180619_180515860    # away from sun
    cap = cv2.VideoCapture('./test_videos/20180619_175221224.mp4')
    #cap = cv2.VideoCapture('./test_videos/Formula Student Spain 2015 Endurance- DHBW Engineering with the eSleek15.mp4')
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    count = 0
    img_number = 1000

    with detection_graph.as_default():
        #with tflow.Session(graph=detection_graph) as sess:
        with tflow.Session(graph=detection_graph, config=tflow.ConfigProto(gpu_options=gpu_options)) as sess:
            while count < frameCount:
                ret, image_np = cap.read()
                if ret == True:
                    count = count + 1
                    # image_np = cv2.resize(processFrame.image, (0,0), fx=0.5, fy=0.5) 
                    #image_np = processFrame.image
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    
                    # Visualization of the results of a detection.
                    # Definition of boxes [ymin, xmin, ymax, xmax]
                    boxes = np.squeeze(boxes)
                    scores = np.squeeze(scores)

                    width = image_np.shape[1]
                    height = image_np.shape[0]
                    # width, height = cv2.GetSize(image_np)
                    output_img = image_np.copy()

                    for i in range(boxes.shape[0]):
                        if np.all(boxes[i] == 0) or scores[i] < threshold_cone:
                            continue
                        
                        b = boxes[i]

                        box_width = np.abs(float(b[3])-float(b[1]))
                        box_height  = np.abs(float(b[2])-float(b[0]))

                        x = int(b[1] * width)
                        y = int(b[0] * height)
                        h = int(box_height * height)
                        w = int(box_width * width)

                        candidate = image_np[y:y+h, x:x+w]

                        # if count % (2*fps) == 0:
                        #     # Save the image (optional)
                        #     cv2.imwrite('./test_videos/cone_samples/' + str(img_number) + '.jpg', candidate)
                        #     img_number = img_number + 1

                        y = y + 1
                        z = 0

                        result = detectCone1(candidate)
                        # print(result)

                        if result == 0:
                            print("Yellow Cone")
                            cv2.rectangle(output_img, (int(b[1] * width),int(b[0] * height)), (x+w,y+h), (0, 255, 255), 7)
                            cv2.putText(output_img, 'yellow cone', (int(b[1] * width),int(b[0] * height)-30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                            cv2.putText(output_img,  str(round(z,1))+" m", (int(b[1] * width),int(b[0] * height)-5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                        if result == 1:
                            print("Blue Cone")
                            cv2.rectangle(output_img, (int(b[1] * width),int(b[0] * height)), (x+w,y+h), (255, 0, 0), 7)
                            cv2.putText(output_img, 'blue cone', (int(b[1] * width),int(b[0] * height)-30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                            cv2.putText(output_img,  str(round(z,1))+" m", (int(b[1] * width),int(b[0] * height)-5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                        if result == 2:
                            print("Orange Cone")
                            cv2.rectangle(output_img, (int(b[1] * width),int(b[0] * height)), (x+w,y+h), (0,165,255), 7)
                            cv2.putText(output_img, 'orange cone', (int(b[1] * width),int(b[0] * height)-30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                            cv2.putText(output_img,  str(round(z,1))+" m", (int(b[1] * width),int(b[0] * height)-5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                    cv2.imshow('object detection', cv2.resize(output_img, (image_np.shape[1],image_np.shape[0])))
                    cv2.waitKey(1)
                    
    cv2.destroyAllWindows()                

if __name__ == '__main__':
    mainLoop()
