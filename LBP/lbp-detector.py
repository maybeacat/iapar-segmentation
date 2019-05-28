import glob, cv2, time
import numpy as np

detect_path = "Projeto_IAPAR/Base_Jersey/sample1/"
detect_path_roi = detect_path + "ROI/"

def cascade_detect(img, cascade, size=200):
    n = 3
    r = cascade.detectMultiScale(img, minNeighbors=n, minSize=(size, size))
    while len(r) > 1:
        n += 1
        r = cascade.detectMultiScale(img, minNeighbors=n, minSize=(size, size))
    return r

# Encontra a boca
def mouth_detector(img):
    #ajuste de contraste
    #im = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #im[:,:,2] = cv2.equalizeHist(im[:,:,2])
    #im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)

    #imagem negativa
    #im = cv2.equalizeHist(img)
    im = 255-img

    #escala de cinza e borrar
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    aux = cv2.GaussianBlur(im, (0, 0), 2)

    #encontra bordas
    median = np.median(aux)
    sigma = 0.2
    lower = 50#int(max(0, (1.0-sigma)*median))
    upper = 100#int(min(255, (1.0+sigma)*median))
    aux = cv2.Canny(aux, lower, upper)

    # fechamento (remove pequenos espaços em branco nas linhas)
    aux = cv2.morphologyEx(aux, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3)))

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
    if z[1] > 0 or z[0] < 0:
        return None

    p = np.poly1d(z)
    return p

# encontra região de interesse (Region Of Interest)
def ROI(image, muzzle, narina_r, narina_l):
    img0 = cv2.imread(image, cv2.IMREAD_COLOR) # imagem original
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) #imagem escala de coinza

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
    if len(right) != 1 or len(left) != 1:
        print('Narina %s não encontrada.'%(
            'direita' if len(right) != 1 else 'esquerda'))
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

    #Coordenadas da região de interesse
    #dentro da imagem do focinho
    w = (250+right[0])-(left[0]+left[2])
    h = w
    x = left[0]+left[2]
    y = (250+menor) - h

    #na imagem original
    x0 = int(x*escala)+m[0]
    y0 = int(y*escala)+m[1]
    x1 = x0+int(w*escala)
    y1 = y0+int(h*escala)

    # Salva região de interesse
    image = image.replace(detect_path, detect_path_roi)
    cv2.imwrite(image, img0[y0:y1, x0:x1])

    # desenha os retangulos nos objetos
    img = cv2.resize(img, (int(500*escala), int(500*escala)))

    #narina esquerda
    x0 = int(left[0]*escala)
    x1 = int(x0+left[2]*escala)
    y0 = int(left[1]*escala)
    y1 = int(y0+left[3]*escala)
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # narina direita
    x0 = int((250+right[0])*escala)
    x1 = int(x0+right[2]*escala)
    y0 = int(right[1]*escala)
    y1 = int(y0+right[3]*escala)
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 0), 2)

    # ROI
    x0 = int(x*escala)
    x1 = int(x0+w*escala)
    y0 = int(y*escala)
    y1 = int(y0+h*escala)
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)

    # exibe a imagem
    print("nice " + image)
    cv2.imwrite("nice_" + image + ".jpg", img)
    
    return 1
#     return focinho[y:y+h, x:x+w]


muzzle = cv2.CascadeClassifier('focinho.xml')
nar_dir = cv2.CascadeClassifier('dir_BR.xml')
nar_esq = cv2.CascadeClassifier('esq_BR.xml')

n = 0
for img in sorted(glob.glob(detect_path + '**/*.jpg', recursive=True)):
    print(img)
    im = ROI(img, muzzle, nar_dir, nar_esq)
    if im != None:
        n += 1

print(n)
