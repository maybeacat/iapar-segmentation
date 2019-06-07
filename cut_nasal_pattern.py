import cv2
import os
import fnmatch
import numpy as np

# command line
import sys
imgs_rootdir = '../imgs/Projeto_IAPAR/Base_Jersey'
if(len(sys.argv) > 1):
    if('-h' in sys.argv or '--help' in sys.argv):
        print('-p (--path)     usa o path que voce quiser, passar como proximo argumento (bom para testar mudancas no codigo com pequenas amostras)')
        print('-h (--help)     mostra essa msg')
        print('')
        print('comportamento padrao:')
        print('- path = ' + imgs_rootdir)
        quit()
    if('-p' in sys.argv or '--path' in sys.argv):
        imgs_rootdir = sys.argv[sys.argv.index('-p')+1 if ('-p' in sys.argv) else sys.argv.index('--path')+1]
        print('custom path mode\n')
print('path: ' + imgs_rootdir)


narinas_file = open("results/YOLO_narinas.csv", "r") # csv onde estao armazenadas as coordenadas das narinas (formato: imgname,x0,y0,w0,h0,x1,y1,w1,h1)
coords_file = open('results/YOLO_coords.csv', 'w') # csv onde serao armazenadas as coordenadas (formato: imgname,x0,y0,x1,y1)

file_counter = 0 # contador de progresso usado em prints
num_files = len(narinas_file.readlines())
narinas_file.seek(0,0)

for linha in narinas_file:
    linha = linha[:-2]
    linha = linha.replace("\n", "")
    linha = linha.split(",")
    
    # procura a linha de nome de arquivo correspondente
    for root, dirnames, filenames in os.walk(imgs_rootdir):
        for filename in fnmatch.filter(filenames, linha[0]):
            imgpath = os.path.join(root, filename)
            break

    # print progresso! arquivos restantes, nome do arquivo, num coords armazenadas (provavelmente multiplos de 4. quatro coords para uma narina...)
    file_counter += 1
    print ("(%.2f%%) %d/%d | %s | n=%d" % ((float(file_counter)/num_files)*100, file_counter, num_files, imgpath.split("/")[-1], len(linha)-1))

    img = cv2.imread(imgpath)

    if len(linha) == 9:
        x1 = int(linha[1])
        y1 = int(linha[2])
        w1 = int(linha[3])
        h1 = int(linha[4])

        x2 = int(linha[5])
        y2 = int(linha[6])
        w2 = int(linha[7])
        h2 = int(linha[8])

        x_esq = min(x1 + w1, x2 + w2)
        x_dir = max(x1, x2)

        ye = y1+0.4*h1
        yd = y2+0.4*h2

        if x1 > x2:
            ye, yd = yd, ye

        # rotação
        if abs(ye-yd) > 5:
            v1 = [1, 0]
            v2 = [x_dir-x_esq, yd-ye]
            dim = int(np.linalg.norm(v2))
            v2 = v2/np.linalg.norm(v2)
            theta = np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1,v2))
            if yd < ye:
                theta = -theta
            ct = np.cos(theta)
            st = np.sin(theta)
            M = np.array([[ct, st, (1-ct)*x_esq - st*ye],
                          [-st, ct, st*x_esq + (1-ct)*ye]])
            sy, sx = np.shape(img)[:2]
            img = cv2.warpAffine(img, M, dsize=(sx, sy))

            y_top = int(ye)
            y_bot = y_top+dim
            x_dir = x_esq+dim
        else:
            dim = x_dir - x_esq
            y_top = int((y1 + 0.4 * h1 + y2 + 0.4 * h2) / 2)
            y_bot = y_top + dim

        if y_top < y_bot and x_esq < x_dir:
            roi = img[y_top : y_bot, x_esq : x_dir]
            cv2.imwrite("results/YOLO_nasalpattern/" + linha[0], roi)
            coords_file.write("%s,%d,%d,%d,%d\n" % (linha[0], x_esq, y_top, x_dir, y_bot))

# fecha arquivos
narinas_file.close()
coords_file.close()
