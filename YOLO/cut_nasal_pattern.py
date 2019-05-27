import cv2

arquivo = open("usp_focinhos.csv", "r")

for linha in arquivo:
	linha = linha[:-2]

	linha = linha.replace("\n", "")
	
	linha = linha.split(",")

	img = cv2.imread("USP/USP/" + linha[0])

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

		dim = x_dir - x_esq

		y_top = int((y1 + 0.4 * h1 + y2 + 0.4 * h2) / 2)
		y_bot = y_top + dim

		if y_top < y_bot and x_esq < x_dir:
			roi = img[y_top : y_bot, x_esq : x_dir]
			cv2.imwrite("USP_nasal_pattern/" + linha[0], roi)
