import cv2
import numpy as np
import fnmatch
import os
import logging
import time

def eqHist(img):
	img_yuv = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

init = time.time()

CONFIDENCE = 0

net = cv2.dnn.readNetFromDarknet("bois-yolov3.cfg", "bois-yolov3_30000.weights")

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

narinas = [0, 0, 0, 0, 0] 
 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.FileHandler('iapar2_focinhos.csv')
handler.setLevel(logging.INFO)

logger.addHandler(handler)
   
# gera um array com todos os paths dos .jpg no diretorio
list_imgs = []
for root, dirnames, filenames in os.walk('../imgs/Projeto_IAPAR/Base_Jersey/~rng'):
    for filename in fnmatch.filter(filenames, '*.jpg'):
        list_imgs.append(os.path.join(root, filename))

# processa as imgs
for addr in list_imgs:
    image = eqHist(cv2.imread(addr))

    if image is None:
        print addr.split("/")[-1], "Nao abriu"
        continue

    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), [0,0,0], swapRB=True, crop=False)

    net.setInput(blob)

    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, 0.3)

    narinas[len(idxs)] += 1

    print addr.split("/")[-1], len(idxs)

    info = addr.split("/")[-1] + ','

    if len(idxs) > 0:
        for i in idxs.flatten():
            for coord in boxes[i]:
                info += str(coord) + ","

    logger.info(info)

    '''
    if len(boxes) == 2:
        direita, esquerda = None, None

        if boxes[0][0] > boxes[1][0]:
            esquerda = boxes[1]
            direita = boxes[0]

        else:
            esquerda = boxes[0]
            direita = boxes[1]

        x_esq = esquerda[0] + esquerda[2]
        x_dir = direita[0]

        dim = x_dir - x_esq

        y_top = (esquerda[1] + esquerda[3] * 0.4 + direita[1] + direita[3] * 0.4) / 2 
        y_bot = y_top + dim 

        logger.info("%s,%s,%s,%s,%s" % (addr.split("/")[-1], x_esq, y_bot, x_dir, y_top))
    '''

    '''
    # ensure at least one detection exists
    if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
     
                    # draw a bounding box rectangle and label on the image
                    color = (0, 255, 0)
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format('narina', confidences[i])
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
     
    # show the output image
    cv2.imwrite("predictions.jpg", image)
    '''

total_time = time.time() - init

print narinas, total_time
