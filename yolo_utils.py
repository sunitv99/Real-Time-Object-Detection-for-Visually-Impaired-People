import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
import pyttsx3 

def convert(mytext):
    engine = pyttsx3.init() 
  
# testing 
    engine.say(mytext) 
    #engine.say("Thank you, Geeksforgeeks") 
    engine.runAndWait() 

def show_image(img):
    cv.imshow("Image", img)
    cv.waitKey(0)

def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels):
    # If there are any detections
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Get the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            
            # Get the unique color for this class
            color = [int(c) for c in colors[classids[i]]]

            # Draw the bounding box rectangle and label on the image
            cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
            text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
            text1 = labels[classids[i]]
            cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            #print(text1)
            convert(text1)

    return img


def generate_boxes_confidences_classids(outs, height, width, tconf):
    boxes = []
    confidences = []
    classids = []
    centers = []


    for out in outs:
        for detection in out:
            #print (detection)
            #a = input('GO!')
            
            # Get the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            
            # Consider only the predictions that are above a certain confidence level
            if confidence > tconf:
                # TODO Check detection
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')

                # Using the center x, y coordinates to derive the top
                # and the left corner of the bounding box
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                # Append to list
                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)
                centers.append((centerX, centerY))
                

    return boxes, confidences, classids, centers

def infer_image(net, layer_names, height, width, img, colors, labels, FLAGS, 
            boxes=None, confidences=None, classids=None, centers=None, idxs=None, infer=True):
    
    if infer:
        # Contructing a blob from the input image
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), 
                        swapRB=True, crop=False)

        # Perform a forward pass of the YOLO object detector
        net.setInput(blob)

        # Getting the outputs from the output layers
        start = time.time()
        outs = net.forward(layer_names)
        end = time.time()

        if FLAGS.show_time:
            print ("[INFO] YOLOv3 took {:6f} seconds".format(end - start))

        
        # Generate the boxes, confidences, and classIDs
        boxes, confidences, classids, centers = generate_boxes_confidences_classids(outs, height, width, FLAGS.confidence)
        
        # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
        idxs = cv.dnn.NMSBoxes(boxes, confidences, FLAGS.confidence, FLAGS.threshold)


        texts = []

		# ensure at least one detection exists
        if len(idxs) > 0:
			# loop over the indexes we are keeping
            for i in idxs.flatten():
				# find positions
                centerX, centerY = centers[i][0], centers[i][1]
				
                if centerX <= width/3:
                    W_pos = "left "
                elif centerX <= (width/3 * 2):
                    W_pos = "center "
                else:
                    W_pos = "right "
				
                if centerY <= height/3:
                    H_pos = "top "
                elif centerY <= (height/3 * 2):
                    H_pos = "mid "
                else:
                    H_pos = "bottom "

                texts.append(H_pos + W_pos + labels[classids[i]])
                print(texts)
                # convert(texts)    

                if texts:
                        description = ', '.join(texts)
                        convert(description)
                    
                
    if boxes is None or confidences is None or idxs is None or classids is None:
        raise '[ERROR] Required variables are set to None before drawing boxes on images.'
        
    # Draw labels and boxes on the image
    img = draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels)

    return img, boxes, confidences, classids, idxs
