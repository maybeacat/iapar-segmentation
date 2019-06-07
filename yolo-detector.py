import cv2
import numpy as np
import time
import fnmatch
import os

# command line
import sys
box_logging = 'normal'
imgs_rootdir = '../imgs/Projeto_IAPAR/Base_Jersey'
if(len(sys.argv) > 1):
    if('-h' in sys.argv or '--help' in sys.argv):
        print('-f (--full)     salva as bounding boxes de TODAS as imgs em /YOLOv3_predictions (requer espaco em disco livre similar ao tamanho da base de entrada)')
        print('-n (--nolog)    nao salva nenhuma bounding box em /YOLOv3_predictions')
        print('-p (--path)     usa o path que voce quiser, passar como proximo argumento (bom para testar mudancas no codigo com pequenas amostras)')
        print('-h (--help)     mostra essa msg')
        print('')
        print('comportamento padrao:')
        print('- salva em /YOLOv3_predictions apenas as bounding boxes das imgs em que nao foram encontradas duas narinas')
        print('- path = ' + imgs_rootdir)
        quit()
    if('-f' in sys.argv or '--full' in sys.argv):
        box_logging = 'full'
        print('full bounding box logging')
    if('-n' in sys.argv or '--nolog' in sys.argv):
        box_logging = 'none'
        print('no bounding box logging')
    if('-p' in sys.argv or '--path' in sys.argv):
        imgs_rootdir = sys.argv[sys.argv.index('-p')+1 if ('-p' in sys.argv) else sys.argv.index('--path')+1]
        print('custom path mode\n')
print('path: ' + imgs_rootdir + '\nbox logging: ' + box_logging + '\n')


# abre arquivos
narinas_file = open('results/YOLO_narinas.csv', 'w') # csv para guardar as bounding boxes das narinas (x1,y1,w1,h1 && x2,y2,w2,h2)
mistakes_file = open('results/YOLO_mistakes.csv', 'w') # csv para guardar os paths das imgs em que nao foram encontradas duas narinas
benchmark_file = open('results/YOLO_benchmark.txt', 'w') # guarda quantidade de narinas encontradas (array 'narinas') e tempo de execucao

# equalizacao de histograma
def eqHist(img):
    img_yuv = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


init = time.time()
CONFIDENCE = 0

# carrega o modelo
net = cv2.dnn.readNetFromDarknet("cfg/bois-yolov3.cfg", "cfg/bois-yolov3_30000.weights")

# pega os nomes das camadas
ln = net.getLayerNames()
# print('net.getLayerNames()')
# print(ln)
# print("")
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]
# print('net.getUnconnectedOutLayers()')
# print(ln)
# print("")

# contador de quantas narinas encontrou para cada animal (normal usar apenas pos 0,1,2)
narinas = [0, 0, 0, 0, 0]

# gera um array com todos os paths dos .jpg no diretorio
list_imgs = []
for root, dirnames, filenames in os.walk(imgs_rootdir):
    for filename in fnmatch.filter(filenames, '*.[Jj][Pp][Gg]'): # case insensitive
        list_imgs.append(os.path.join(root, filename))

# processa as imgs
file_counter = 0 # contador de progresso usado em prints

for addr in list_imgs:
    image = eqHist(cv2.imread(addr))

    if image is None:
        print (addr.split("/")[-1], "Nao abriu")
        continue

    (H, W) = image.shape[:2]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), [0,0,0], swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs and each of the detections
    for output in layerOutputs:
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, 0.3)


    # armazena o numero de narinas encontradas
    narinas[len(idxs)] += 1
    
    # salva os casos de falha (narinas != 2) em um arquivo
    if(len(idxs) != 2):
        mistakes_file.write('%s,%d\n' % (addr, len(idxs)))


    # print progresso! arquivos restantes, nome do arquivo, narinas encontradas
    file_counter += 1
    print ("(%.2f%%) %d/%d | %s | n=%d" % ((float(file_counter)/len(list_imgs))*100, file_counter, len(list_imgs), addr.split("/")[-1], len(idxs)))

    info = addr.split("/")[-1] + ','


    # salva 
    if len(idxs) > 0:   # ensure at least one detection exists
        for i in idxs.flatten():    # loop over the indexes we are keeping
            for coord in boxes[i]:
                info += str(coord) + ","
                
    narinas_file.write(info + '\n')


    # parte que salva a img com bounding boxes (dependendo do nivel de logging)
    if box_logging != 'none':
        if len(idxs) > 0:
            if(box_logging == 'full' or len(idxs) != 2):
                for i in idxs.flatten():  # loop over the indexes we are keeping
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # draw a bounding box rectangle and label on the image
                    color = (255, 0, 255)
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 8)
                    text = "{}: {:.2f}".format('narina', confidences[i])
                    cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_TRIPLEX, 3, color, 2) # desenha texto
                    imgr = cv2.resize(image, None, fx=0.2, fy=0.2) # redimensiona para ocupar menos espaco
                    cv2.imwrite("results/YOLO_predictions/" + addr.split("/")[-1], imgr) # salva img

# calcula o tempo que demorou para terminar tudo
total_time = time.time() - init

# benchmarking
print()
benchmark_str = "Detected: %d/%d (%.2f)\nNarinas: [%d,%d,%d,%d,%d]\nTempo: %.2f" % (narinas[2], len(list_imgs), (float(narinas[2])/len(list_imgs))*100, narinas[0], narinas[1], narinas[2], narinas[3], narinas[4], total_time)
print(benchmark_str)
benchmark_file.write(benchmark_str)

# fecha arquivos
narinas_file.close()
mistakes_file.close()
benchmark_file.close()
