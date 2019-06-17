# baseado em https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

# import the necessary packages
from collections import namedtuple
import numpy as np
import cv2

# define the `Detection` object
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


detector_file = open('results/YOLO_coords.csv', 'r') # csv onde estao armazenadas as coordenadas geradas pelo detector (formato: imgname,x0,y0,x1,y1)
gt_file = open('results/gt-IAPAR2_3.csv', 'r') # csv onde estao armazenadas as coordenadas ground truth (formato: imgname,x0,y0,x1,y1)

# prepara listas de coordenadas do detector e ground truth
detector_list = []
gt_list = []

# listas de nomes para ficar mais facil de pegar os index
detector_names = []
gt_names = []

for linha in detector_file:
    linha = linha.strip().split(",")
    detector_list.append(linha)
    detector_names.append(linha[0])

for linha in gt_file:
    linha = linha.strip().split(",")
    gt_list.append(linha)
    gt_names.append(linha[0])
    
# procura matches
matches = 0  # numero de correspondencias nos dois arquivos
i = 0
while i < len(detector_names):
    try:
        gti = gt_names.index(detector_names[i])  # ground truth index
        
        # transforma strings em ints
        argdec = [int(e) for e in detector_list[i][1:]]
        arggt = [int(e) for e in gt_list[gti][1:]]
        print("IOU %s: %.3f" % (detector_names[i], bb_intersection_over_union(argdec, arggt)))
        matches += 1
    except ValueError:
        matches += 0
        # print("faltando match para ", detector_names[i])
    i += 1
print(matches)
