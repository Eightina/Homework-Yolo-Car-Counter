# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python car_counter_yolo.py --video=test1.mp4 --device 'cpu' --time_=day --sampling=1
#                 python car_counter_yolo.py --video=test2.mp4 --device 'cpu' --time_=night --sampling=1
#                 python car_counter_yolo.py --video=run.mp4 --device 'gpu'
#                 python car_counter_yolo.py --image=bird.jpg --device 'cpu'
#                 python car_counter_yolo.py --image=bird.jpg --device 'gpu'

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import point_in_poly as poinpo
import torch 
import csv



# Initialize the parameters
confThreshold = 0.3  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--device', default='cpu', help="Device to perform inference on 'cpu' or 'gpu'.")
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.', default='test1.mp4')
parser.add_argument('--time_', help='Time of the video.', default='day')
parser.add_argument('--sampling', help='Frequency of sampling.', default=1)
args = parser.parse_args()
        
# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

if(args.device == 'cpu'):
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    print('Using CPU device.')
elif(args.device == 'gpu'):
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    print('Using GPU device.')
sampling = int(args.sampling)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#############################preload####################################
#############################preload####################################
#############################preload####################################

# Get lane coordinates for different video.
def getCo(videoTime):
    if videoTime == 'day':
        lane1 = [[(752.3653624856158, 567.9958079894789), 
                    (660.6343087292455, 656.7677954956436), 
                    (409.1136774617788, 857.9843005096169), 
                    (133.92051619266806, 1071.0370705244122), 
                    (361.7686174584909, 1073.9961367746178), 
                    (554.1079237218478, 869.8205655104389), 
                    (728.6928324839719, 659.7268617458491), 
                    (793.7922899884927, 567.9958079894789),
                    (752.3653624856158, 567.9958079894789)]]
        lane2 = [[(802.669488739109, 562.0776754890679),
                    (719.8156337333553, 686.3584579976985),
                    (557.0669899720533, 872.7796317606444),
                    (388.4002137103404, 1071.0370705244122),
                    (634.0027124773962, 1073.9961367746178),
                    (728.6928324839719, 881.6568305112609),
                    (805.6285549893146, 689.317524247904),
                    (844.0964162419859, 562.0776754890679),
                    (802.669488739109, 562.0776754890679)]]
        lane3 = [[(850.014548742397, 562.0776754890679),
                    (817.4648199901367, 680.4403254972875),
                    (734.6109649843829, 860.9433667598224),
                    (642.8799112280126, 1068.0780042742067),
                    (888.4824099950683, 1068.0780042742067),
                    (900.3186749958904, 860.9433667598224),
                    (906.2368074963013, 683.3993917474929),
                    (888.4824099950683, 562.0776754890679),
                    (850.014548742397, 562.0776754890679)]]
        lane4 = [[(897.3596087456849, 562.0776754890679),
                    (915.1140062469178, 650.8496629952326),
                    (923.9912049975344, 831.3527042577675),
                    (926.9502712477397, 1073.9961367746178),
                    (1181.429968765412, 1073.9961367746178),
                    (1066.026385007398, 825.4345717573565),
                    (986.1315962518497, 647.8905967450271),
                    (941.7456024987673, 565.0367417392733),
                    (897.3596087456849, 562.0776754890679)]]
        lane5 = [[(947.6637349991781, 565.0367417392733),
                    (1009.8041262534935, 674.5221929968765),
                    (1095.6170475094527, 866.8614992602334),
                    (1181.429968765412, 1071.0370705244122),
                    (1418.1552687818514, 1071.0370705244122),
                    (1246.5294262699329, 860.9433667598224),
                    (1092.6579812592472, 671.5631267466711),
                    (989.090662502055, 567.9958079894789),
                    (947.6637349991781, 565.0367417392733)]]
    elif videoTime == 'night':
        lane1 = [[(808.5876212395201, 502.896350484958),
                (699.1021699819169, 606.4636692421502),
                (474.21313496629955, 792.8848430050962),
                (148.71584744369557, 1068.0780042742067),
                (382.48208120992933, 1073.9961367746178),
                (634.0027124773962, 795.8439092553017),
                (776.0378924872598, 609.4227354923557),
                (847.0554824921915, 502.896350484958),
                (808.5876212395201, 502.896350484958)]]
        lane2 = [[(852.9736149926025, 502.896350484958), 
                (784.9150912378761, 609.4227354923557), 
                (622.1664474765741, 813.5983067565346), 
                (403.1955449613678, 1068.0780042742067), 
                (672.4705737300676, 1068.0780042742067),
                (784.9150912378761, 813.5983067565346),
                (864.8098799934244, 612.3818017425613),
                (894.4005424954794, 499.9372842347526),
                (852.9736149926025, 502.896350484958)]]
        lane3 = [[(897.3596087456849, 502.896350484958),
                (864.8098799934244, 624.2180667433831),
                (790.8332237382872, 807.6801742561236),
                (666.5524412296566, 1068.0780042742067),
                (944.7046687489728, 1068.0780042742067),
                (947.6637349991781, 801.7620417557126),
                (953.5818674995892, 621.2590004931777),
                (938.7865362485618, 499.9372842347526),
                (897.3596087456849, 502.896350484958)]]
        lane4 = [[(947.6637349991781, 496.9782179845471), 
                (962.4590662502058, 582.7911392405063),
                (974.2953312510276, 849.1071017590004),
                (977.2543975012331, 1068.0780042742067),
                (1258.3656912707547, 1068.0780042742067),
                (1160.7165050139736, 849.1071017590004), 
                (1039.3947887555485, 585.7502054907118), 
                (983.1725300016442, 496.9782179845471),
                (947.6637349991781, 496.9782179845471)]]
        lane5 = [[(989.090662502055, 496.9782179845471),
                (1057.1491862567814, 624.2180667433831),
                (1140.0030412625351, 801.7620417557126),
                (1258.3656912707547, 1073.9961367746178), 
                (1524.6816537892491, 1073.9961367746178), 
                (1299.7926187736316, 798.8029755055071), 
                (1151.8393062633572, 618.2999342429722), 
                (1024.599457504521, 482.1828867335196),
                (989.090662502055, 496.9782179845471)]]
    return [lane1, lane2, lane3, lane4, lane5]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
    
def inLane(judge_x, judge_y, laneArray, videoTime):
    inLaneArray=[]
    objInLane=999
    lane = getCo(videoTime)
    for i in range(5):
        res=poinpo.is_point_in_polygon((judge_x, judge_y), lane[i], False)
        inLaneArray.append(res)
    for i in inLaneArray:
        if i == 1:
            idx = inLaneArray.index(i)
            laneArray[idx] += 1
            objInLane = idx+1
    return laneArray,objInLane

cnt=0
carCount=[0,0,0,0,0]
idDic={2:'car', 5:'bus', 7:'truck'}
counted=[]

# Track, recognize each object and give it a number by comparing coordinates of objects between the former frame and the new frame
def tracker(preObj, newObj, laneArray, videoTime):
    # print('preObj=',preObj)
    # print('newObj=',newObj)
    global cnt
    global carCount
    global counted
    isNew=True
    # Making comparison betwenn preObj and newObj
    for n in newObj:
        idxN = newObj.index(n)
        for p in preObj:
            idxP = preObj.index(p)
            dis=torch.norm(torch.tensor([n[2]-p[2],n[3]-p[3]]))
            # Update coordinates of already recognized objects  
            # print('dis=',dis)
            if dis<(n[3]*65/1080):
                n[1]=p[1]
                isNew = False
                break
            isNew = True
        # Add newly recognized objects
        if isNew == True:
            cnt+=1
            n[1]=cnt
    objInfo = newObj[:]
    
    for o in objInfo:
        o[0]=idDic[o[0]]
    # Structure of objInfo: [[classId, objNum, x_judge, y_judge],...]
    
    # Count vehicles within a limitied area 
    for n in newObj:
        idxN = newObj.index(n)
        for p in preObj:
            idxP = preObj.index(p)
            if n[1]==p[1] and ((p[3]-775)*(n[3]-775)) <= 0 and (n[1] not in counted):
                laneArray, objInLane = inLane(n[2], n[3], laneArray, videoTime)
                carCount[objInLane-1]+=1
                counted.append(n[1])
                newObj.remove(n)
                preObj.remove(p)                   
    return objInfo
    
    
    
    
    
# Remove the bounding boxes with low confidence using non-maxima suppression
# Record numbers of vehicles in each lane as laneArray
# Track objects in defined area with tracker()
# Return information of recognized & tracked objects in each frame as preObj
def postprocess(frame, outs, laneArray, preObj, vieoTime):
    global frameNum
    global f1Cont
    global f2Cont
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    selectedClassId = [2,5,7]
    newObj=[]
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            # Find the most possible class
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold and classId in selectedClassId:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        x = left + width/2
        if x < frameWidth/2:
            judge_x = (left + width/4)
        else:
            judge_x = (left + width*3/4)
        judge_y = top + height
        laneArray,objInLane = inLane(judge_x, judge_y, laneArray, vieoTime)
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
        f1Cont.append([str(frameNum),idDic[classIds[i]],str(confidences[i]),'('+str(left)+','+str(top)+')','('+str(left+width)+','+str(judge_y)+')'])
        # ['frameNum', 'objClass', 'objConf', 'lefttop', 'rightbottom' ]
        # Build the set of objects in the new frame
        if judge_y>=725 and judge_y<=800:
            newObj.append([classIds[i], -1, judge_x, judge_y])

    # Get info to write in csv       
    objInfo=tracker(preObj, newObj, laneArray, vieoTime)
    for o in objInfo:
        classId = o[0]
        objNum = o[1]
        x=o[2]
        y=o[3]
        f2Cont.append([str(frameNum),str(objNum),classId,x,y])
        
    #structure of objInfo: [[classId, objNum, x_judge, y_judge],...] 
    #headers2 = ['frameNum', 'objNum', 'objClass', 'x', 'y']
    #print('objInfo:',objInfo)
    preObj=[]        
    preObj=objInfo[:]
    for o in objInfo:
        cv.putText(frame, o[0]+str(o[1]), (int(o[2]), int(o[3])-10), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    return preObj

            

# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    # Establish reading obj.
    outputFile = args.video[:-4]+'_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
preObj = []
frameNum = 0
f1Cont = []
f2Cont = []
while cv.waitKey(1) < 0:
    frameNum += 1
    if frameNum%sampling == 0:
        laneArray = [0, 0, 0, 0, 0]
        # get frame from the video
        hasFrame, frame = cap.read()
        
        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            cv.waitKey(3000)
            # Release device
            cap.release()
            # Write csv files
            headers1 = ['frameNum', 'objClass', 'objConf', 'lefttop', 'rightbottom' ]
            headers2 = ['frameNum', 'objNum', 'objClass','trackingX', 'trackingY']
            filename1 = 'allObj.csv'
            filename2 = 'trackedObj.csv'
            with open(filename1,'w',newline='')as f1:
                f1csv = csv.writer(f1)
                f1csv.writerow(headers1)
                f1csv.writerows(f1Cont)
            with open(filename2,'w',newline='')as f2:
                f2csv = csv.writer(f2)
                f2csv.writerow(headers2)
                f2csv.writerows(f2Cont)
            break

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence and track objects
        preObj = postprocess(frame, outs, laneArray, preObj, args.time_)
        
        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        # Put statistcs of road traffic, including total count and realtime count
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv.putText(frame, 'Count:', (0, 900), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255),2)
        cv.putText(frame, 'realtimeObj:'+str(laneArray) , (0, 700), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255),2)
        for i in range(5):
            cv.putText(frame, 'lane'+str(i+1)+':'+str(carCount[i]), (200+(i+1)*200, 900), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255),2)
        # Write the frame with the detection boxes
        if (args.image):
            cv.imwrite(outputFile, frame.astype(np.uint8s))
        else:
            vid_writer.write(frame.astype(np.uint8))

        cv.imshow(winName, frame)



