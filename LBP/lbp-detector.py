import cv2
import numpy as np
import time
import fnmatch
import os

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


# abre arquivos
narinas_file = open('results/LBP_narinas.csv', 'w') # csv para guardar as bounding boxes das narinas (x1,y1,w1,h1 && x2,y2,w2,h2)
coords_file = open('results/LBP_coords.csv', 'w') # csv onde serao armazenadas as coordenadas (formato: imgname,x0,y0,x1,y1)
mistakes_file = open('results/LBP_mistakes.csv', 'w') # csv para guardar os paths das imgs em que nao foram encontradas duas narinas
benchmark_file = open('results/LBP_benchmark.txt', 'w') # guarda quantidade de narinas encontradas (array 'narinas') e tempo de execucao

# contador de quais narinas foram encontradas [L, R]. idealmente o resultado seria [n, n] (n = numero de imgs)
narinas = [0, 0]

def cascade_detect(img, cascade, size=200):
    n = 3
    r = cascade.detectMultiScale(img, minNeighbors=n, minSize=(size, size))
    while len(r) > 1:
        n += 1
        r = cascade.detectMultiScale(img, minNeighbors=n, minSize=(size, size))
    return r


# Encontra a boca
def mouth_detector(img):
    # ajuste de contraste
    #im = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #im[:,:,2] = cv2.equalizeHist(im[:,:,2])
    #im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)

    # imagem negativa
    #im = cv2.equalizeHist(img)
    im = 255-img

    # escala de cinza e borrar
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    aux = cv2.GaussianBlur(im, (0, 0), 2)

    # encontra bordas
    median = np.median(aux)
    sigma = 0.2
    lower = 50#int(max(0, (1.0-sigma)*median))
    upper = 100#int(min(255, (1.0+sigma)*median))
    aux = cv2.Canny(aux, lower, upper)

    # fechamento (remove pequenos espaços em branco nas linhas)
    aux = cv2.morphologyEx(aux, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3)))

    # encontra as bordas
    contours, hierarchy = cv2.findContours(aux, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # encontra a maior borda horizontalmente
    tam = 0
    for c in contours:
        maxv = np.max(c[:,0,0])
        minv = np.min(c[:,0,0])
        if maxv-minv > tam:
            tam = maxv-minv
            maior = c

    if tam == 0:
        return None

    # função da linha
    z = np.polyfit(maior[:,0,0], maior[:,0,1], 2)

    # COMENTAR AS 2 LINHAS ABAIXO PARA VISUALIZAÇÃO DAS LINHAS ERRADAS
    # if z[1] > 0 or z[0] < 0:
    #    return None

    p = np.poly1d(z)
    return p

# encontra região de interesse (Region Of Interest)
def ROI(addr, muzzle, narina_r, narina_l):
    img0 = cv2.imread(addr, cv2.IMREAD_COLOR) # imagem original
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) # imagem escala de cinza

    # procura focinho
    m = cascade_detect(gray, muzzle)
    # m = lista dos pontos focinhos encontrados
    # m[i] = (x, y, w, h) do i-ésimo focinho
    # x, y = posição do ponto inicial
    # w, h = largura e altura
    if len(m) != 1:
        print('Focinho não encontrado.')
        return None

    m = m[0]
    escala = m[2]/500
    #print('m2: ', m[2], 'escala: ', escala)
    focinho = cv2.resize(gray[m[1]:m[1]+m[3], m[0]:m[0]+m[2]], (500, 500))
    img = cv2.resize(img0[m[1]:m[1]+m[3], m[0]:m[0]+m[2]], (500, 500))

    # procura narinas nos cantos superiores do focinho
    right = cascade_detect(focinho[:,250:], narina_r, 90)
    left = cascade_detect(focinho[:,:250], narina_l, 90)
    
    # contar narinas encontradas. se nao forem duas, salvar caso de falha e abortar imagem
    narinas[0] += len(left)
    narinas[1] += len(right)
    if len(left) == 0 or len(right) == 0:  # quero passar por todos os ifs
        if len(left) == 0:
            print('Narina esquerda não encontrada.')
        if len(right) == 0:
            print('Narina direita não encontrada.')
        mistakes_file.write('%s,[%d %d]\n' % (addr, len(right), len(left)))
        return None

    left = left[0]
    right = right[0]

    # procura linha da boca na metade inferior da imagem
    p = mouth_detector(focinho[250:])#, left[0]:250+right[0]+right[2]])#focinho[250:])
    if not p:
        print('Boca não identificada.')
        return None

    # NÃO UTILIZADO: verifica a inclinação da imagem
    #pl = p(left[0]+(left[2]/2))
    #pr = p((250+right[0])+(right[2]/2))
    #if abs(pl-pr) > 10:
    #    print('Rotaciona')

    # Desenha a linha roxa na imagem e encontra o ponto mais alto da boca
    menor = 250
    for i in range(int(left[0]+(left[2]/2)), int((250+right[0])+(right[2])/2)):
        yp = int(p(i))
        try:
            img[249+yp:251+yp, i] = (255, 0, 255)
        except:
            continue
        if yp < menor:
            menor = yp

    # Coordenadas da região de interesse
    # dentro do corte do focinho
    w = (250+right[0])-(left[0]+left[2])
    h = w
    x = left[0]+left[2]
    y = (250+menor) - h

    # na imagem original
    x0 = int(x*escala)+m[0]
    y0 = int(y*escala)+m[1]
    x1 = x0+int(w*escala)
    y1 = y0+int(h*escala)

    # salva região de interesse
    cv2.imwrite("results/LBP_nasalpattern/" + addr.split("/")[-1], img0[y0:y1, x0:x1])

    # desenha os retangulos nos objetos e salva as coordenadas
    # narina esquerda
    x0e = int(m[0] + left[0]*escala)
    x1e = int(x0e + left[2]*escala)
    y0e = int(m[1] + left[1]*escala)
    y1e = int(y0e + left[3]*escala)
    cv2.rectangle(img0, (x0e, y0e), (x1e, y1e), (0, 255, 0), 2)
    # narina direita
    x0d = int(m[0] + x*escala + w*escala)
    x1d = int(x0d + right[2]*escala)
    y0d = int(m[1] + right[1]*escala)
    y1d = int(y0d + right[3]*escala)
    cv2.rectangle(img0, (x0d, y0d), (x1d, y1d), (255, 255, 0), 2)
    narinas_file.write("%s,%d,%d,%d,%d,%d,%d,%d,%d\n" % (addr.split("/")[-1], x0e,y0e,x1e-x0e,y1e-y0e, x0d,y0d,x1d-x0d,y1d-y0d)) # (x1,y1,w1,h1 && x2,y2,w2,h2)
    # ROI
    x0 = int(m[0] + x*escala)
    x1 = int(x0 + w*escala)
    y0 = int(m[1] + y*escala)
    y1 = int(y0 + h*escala)
    cv2.rectangle(img0, (x0, y0), (x1, y1), (0, 0, 255), 2)
    coords_file.write("%s,%d,%d,%d,%d\n" % (addr.split("/")[-1], x0, y0, x1, y1))

    # salva a imagem
    cv2.imwrite("results/LBP_predictions/" + addr.split("/")[-1], img0)
    
    return focinho[y:y+h, x:x+w]

init = time.time()

muzzle = cv2.CascadeClassifier('cfg/focinho.xml')
nar_dir = cv2.CascadeClassifier('cfg/dir_BR.xml')
nar_esq = cv2.CascadeClassifier('cfg/esq_BR.xml')


# gera um array com todos os paths dos .jpg no diretorio
list_imgs = []
for root, dirnames, filenames in os.walk(imgs_rootdir):
    for filename in fnmatch.filter(filenames, '*.jpg'):
        list_imgs.append(os.path.join(root, filename))

file_counter = 0 # contador de progresso usado em prints
n_success = 0 # numero de ROI detectadas
for addr in list_imgs:
    file_counter += 1
    print ("(%.2f%%) %d/%d | %s" % ((float(file_counter)/len(list_imgs))*100, file_counter, len(list_imgs), addr.split("/")[-1]))
    img = ROI(addr, muzzle, nar_dir, nar_esq)
    if img is not None:
        n_success += 1

# calcula o tempo que demorou para terminar tudo
total_time = time.time() - init

# benchmarking
print()
benchmark_str = "Detected: %d/%d (%.2f)\nNarinas: [%d,%d]\nTempo: %.2f" % (n_success, len(list_imgs), (float(n_success)/len(list_imgs))*100, narinas[0], narinas[1], total_time)
print(benchmark_str)
benchmark_file.write(benchmark_str)

# fecha arquivos
narinas_file.close()
coords_file.close()
mistakes_file.close()
benchmark_file.close()
