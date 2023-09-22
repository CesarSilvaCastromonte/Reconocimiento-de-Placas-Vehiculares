import cv2
import pytesseract
import sys

# Configura la ruta al ejecutable de Tesseract OCR (ajusta esta ruta según tu instalación)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\...\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

image = cv2.imread(r'C:\Users\...\Desktop\MACHINE LEARNING\RECONOCIMIENTO DE PLACAS\auto001.jpg')

# Comprueba si la imagen se cargó correctamente
if image is None:
    print("No se pudo cargar la imagen.")
    sys.exit(1)  # Termina el script con un código de error

# Convierte la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplica un desenfoque para mejorar la detección de bordes
gray = cv2.blur(gray, (3, 3))

# Detecta bordes en la imagen
canny = cv2.Canny(gray, 150, 200)
canny = cv2.dilate(canny, None, iterations=1)

# Encuentra los contornos en la imagen
cnts, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    epsilon = 0.09 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    if len(approx) == 4 and area > 9000:
        print('area=', area)

        # Recorta la región de la placa
        placa = gray[y:y+h, x:x+w]

        # Realiza el reconocimiento de texto en la placa
        text = pytesseract.image_to_string(placa, config='--psm 11')
        print('PLACA: ', text)

        # Muestra la región de la placa en una ventana
        cv2.imshow('PLACA', placa)
        cv2.moveWindow('PLACA', 780, 10)

        # Dibuja un rectángulo alrededor de la placa en la imagen original (color azul)
        #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)  # Cambia el color a azul (BGR)
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Cambia el color a rojo (BGR)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 3)  # Cambia el color a blanco (BGR)BALNCO

        # Aumenta el tamaño de la letra
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1.5, 2)[0]  # Ajusta el tamaño a 1.5
        text_x = x + (w - text_size[0]) // 2
        text_y = y + h + text_size[1] + 10  # Ajusta la posición vertical del texto
        #cv2.putText(image, text, (text_x, text_y), font, 1.5, (255, 0, 0), 2)  # Cambia el tamaño a 1.5 (BGR)AZUL
        #cv2.putText(image, text, (text_x, text_y), font, 1.5, (0, 0, 255), 2)  # Cambia el tamaño a 1.5 (BGR)ROJO
        cv2.putText(image, text, (text_x, text_y), font, 1.5, (255, 255, 255), 2)  # Cambia el tamaño a 1.5 (BGR)BLANCO
        

# Muestra la imagen original con los resultados
cv2.imshow('Image', image)
cv2.moveWindow('Image', 45, 10)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Termina el script sin mostrar el signo de pregunta al final
sys.exit(0)
