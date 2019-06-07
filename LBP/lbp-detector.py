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
benchmark_file = open('results/LBP_benchmark.txt', 'w') # guarda quantidade de focinhos e narinas encontradas (array 'narinas') e tempo de execucao

# contador de quantos focinhos e narinas foram encontrados [F, L, R]. idealmente o resultado seria [n, n, n] (n = numero de imgs)
narinas = [0, 0, 0]


def cascade_detect(img, cascade, size=200):
    n = 3
    r = cascade.detectMultiScale(img, minNeighbors=n, minSize=(size, size))
    while len(r) > 1:
        n += 1
        r = cascade.detectMultiScale(img, minNeighbors=n, minSize=(size, size))
    return r


# python2 gosta de reclamar de acentos no codigo...
# encontra regiao de interesse (Region Of Interest)
def ROI(addr, muzzle, narina_r, narina_l):
    img0 = cv2.imread(addr, cv2.IMREAD_COLOR)  # imagem original
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
    img0[:,:,2] = cv2.equalizeHist(img0[:,:,2])
    img0 = cv2.cvtColor(img0, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)  # imagem escala de cinza

    ty, tx = np.shape(gray) # tamanho da imagem

    gray = cv2.equalizeHist(gray)

    # CLAHE - leve piorada na performance comparando com equalizar normal?
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # gray = clahe.apply(gray)

    # procura focinho
    m = cascade_detect(gray, muzzle, int(min(tx, ty)/3))
    # m = lista dos pontos focinhos encontrados
    # m[i] = (x, y, w, h) do i-esimo focinho
    # x, y = posicao do ponto inicial
    # w, h = largura e altura

    if len(m) != 1:
        print('Focinho nao encontrado.')
        mistakes_file.write('%s,[%d %d %d]\n' % (addr, 0, 0, 0))
        return None

    narinas[0] += 1 # focinho detectado

    m = m[0]
    dy = m[3]-m[1]
    dx = m[2]-m[0]
    dy = int(dy/20)
    dx = int(dx/20)

    escala = (m[2]+2*dx) / 500

    if m[1]-dy < 0:
        y0 = 0
        yf = m[3]+2*dy
    elif m[1]+m[3]+dy >= ty:
        yf = ty-1
        y0 = yf-m[3]-2*dy
    else:
        y0 = m[1]-dy
        yf = m[1]+m[3]+dy

    if m[0]-dx < 0:
        x0 = 0
        xf = m[2]+2*dx
    elif m[0]+m[2]+dx >= tx:
        xf = tx-1
        x0 = xf-m[2]-2*dx
    else:
        x0 = m[0]-dx
        xf = m[0]+m[2]+dx

    focinho = cv2.resize(gray[y0:yf, x0:xf], (500, 500))
    img = cv2.resize(img0[y0:yf, x0:xf], (500, 500))

    # procura narinas nos cantos superiores do focinho
    right = cascade_detect(focinho[:, 250:], narina_r, 75)
    left = cascade_detect(focinho[:, :250], narina_l, 75)
    
    # contar narinas encontradas. se nao forem duas, salvar caso de falha e abortar imagem
    narinas[1] += len(left)
    narinas[2] += len(right)
    if len(left) == 0 or len(right) == 0:  # quero executar write e return depois de ambos os ifs
        if len(left) == 0:
            print('Narina esquerda nao encontrada.')
        if len(right) == 0:
            print('Narina direita nao encontrada.')
        mistakes_file.write('%s,[%d %d %d]\n' % (addr, 1, len(right), len(left)))
        return None

    left = left[0]
    right = right[0]

    # Coordenadas da regiao de interesse
    # dentro da imagem do focinho
    w = (250 + right[0]) - (left[0] + left[2]) # distancia entre narinas
    h = w
    x = left[0] + left[2]
    y = int(((left[1]+left[3]/2)+(right[1]+right[3]/2))/2)

    # aw = int(w*0.15)
    # ah = int(h*0.15)
    # x -= aw
    # y -= ah
    # w += 2*aw
    # h += 2*ah

    # na imagem original
    x0 = int(x * escala) + m[0]
    y0 = int(y * escala) + m[1]
    x1 = x0 + int(w * escala)
    y1 = y0 + int(h * escala)

    # salva regiao de interesse
    cv2.imwrite("results/LBP_nasalpattern/" + addr.split("/")[-1], img0[y0:y1, x0:x1])

    # desenha os retangulos nos objetos e salva as coordenadas
    # narina esquerda
    x0e = int(m[0] + left[0]*escala)
    x1e = int(x0e + left[2]*escala)
    y0e = int(m[1] + left[1]*escala)
    y1e = int(y0e + left[3]*escala)
    cv2.rectangle(img0, (x0e, y0e), (x1e, y1e), (0, 255, 0), 8)
    # narina direita
    x0d = int(m[0] + x*escala + w*escala)
    x1d = int(x0d + right[2]*escala)
    y0d = int(m[1] + right[1]*escala)
    y1d = int(y0d + right[3]*escala)
    cv2.rectangle(img0, (x0d, y0d), (x1d, y1d), (255, 255, 0), 8)
    narinas_file.write("%s,%d,%d,%d,%d,%d,%d,%d,%d\n" % (addr.split("/")[-1], x0e,y0e,x1e-x0e,y1e-y0e, x0d,y0d,x1d-x0d,y1d-y0d)) # (x1,y1,w1,h1 && x2,y2,w2,h2)
    # ROI
    x0 = int(m[0] + x*escala)
    x1 = int(x0 + w*escala)
    y0 = int(m[1] + y*escala)
    y1 = int(y0 + h*escala)
    cv2.rectangle(img0, (x0, y0), (x1, y1), (0, 0, 255), 8)
    coords_file.write("%s,%d,%d,%d,%d\n" % (addr.split("/")[-1], x0, y0, x1, y1))

    # redimensiona e salva a imagem (salva todas... redimensiona para ocupar menos espaco)
    imgr = cv2.resize(img0, None, fx=0.2, fy=0.2)
    cv2.imwrite("results/LBP_predictions/" + addr.split("/")[-1], imgr)
    
    return focinho[y:y+h, x:x+w]

# end defs. comeco main
init = time.time()

muzzle = cv2.CascadeClassifier('cfg/focinho.xml')
nar_dir = cv2.CascadeClassifier('cfg/dir_BR.xml')
nar_esq = cv2.CascadeClassifier('cfg/esq_BR.xml')


# gera um array com todos os paths dos .jpg no diretorio
list_imgs = []
for root, dirnames, filenames in os.walk(imgs_rootdir):
    for filename in fnmatch.filter(filenames, '*.[Jj][Pp][Gg]'): # case insensitive
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
benchmark_str = "Detected: %d/%d (%.2f)\nNarinas: [%d,%d,%d]\nTempo: %.2f" % (n_success, len(list_imgs), (float(n_success)/len(list_imgs))*100, narinas[0], narinas[1], narinas[2], total_time)
print(benchmark_str)
benchmark_file.write(benchmark_str)

# fecha arquivos
narinas_file.close()
coords_file.close()
mistakes_file.close()
benchmark_file.close()
