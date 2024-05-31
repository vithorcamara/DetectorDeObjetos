import cv2
import numpy as np
import tkinter as tk
from tkinter import Button, filedialog
from PIL import Image, ImageTk
import os

# Defina os caminhos absolutos para os arquivos necessários
base_path = os.path.dirname(__file__)
labels_path = os.path.join(base_path, 'coco.names')
config_path = os.path.join(base_path, 'yolov3.cfg')
weights_path = os.path.join(base_path, 'yolov3.weights')

# Função para carregar nomes das classes
def get_classes(file):
    with open(file, 'r') as f:
        classes = f.read().strip().split("\n")
    return classes

# Função para ligar e desligar a câmera
def toggle_camera():
    global cap, btn_toggle_camera, is_running
    if cap is None:
        cap = cv2.VideoCapture(0)
        # Definir a resolução desejada (exemplo: 1280x720)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        btn_toggle_camera.config(text="Desligar Câmera")
        is_running = True
        update_camera()
    else:
        is_running = False
        cap.release()
        cap = None
        btn_toggle_camera.config(text="Ligar Câmera")

def choose_input_mode(mode):
    global cap, btn_toggle_camera, is_running
    if cap is not None:
        is_running = False
        cap.release()  # Libera a captura de vídeo atual, se houver
        btn_toggle_camera.config(text="Ligar Câmera")
        btn_choose_input.config(text="Selecionar Vídeo")

    if mode == 'webcam':
        cap = cv2.VideoCapture(0)  # Inicia a captura da webcam
        btn_toggle_camera.config(text="Desligar Câmera")
    elif mode == 'video':
        # Abrir uma janela de diálogo para escolher um arquivo de vídeo
        video_file = filedialog.askopenfilename(initialdir="/", title="Selecione um arquivo de vídeo",
                                                filetypes=(("Arquivos de Vídeo", "*.mp4;*.avi;*.mov"), ("Todos os arquivos", "*.*")))
        if video_file:  # Verifica se um arquivo foi selecionado
            cap = cv2.VideoCapture(video_file)  # Inicia a captura do arquivo de vídeo selecionado
            btn_choose_input.config(text="Parar Vídeo")
        else:
            cap = None
            return

    is_running = True
    update_camera()  # Atualiza o quadro da câmera na interface

# Função para redimensionar a imagem mantendo a proporção
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# Função para atualizar o frame da câmera na interface
def update_camera():
    global cap, panel, net, classes, is_running
    if is_running and cap is not None:  # Check if the camera is initialized
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = resize_with_aspect_ratio(frame, width=720)

            # Detectar objetos usando YOLO
            (H, W) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            ln = net.getLayerNames()
            ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
            layer_outputs = net.forward(ln)

            boxes = []
            confidences = []
            class_ids = []

            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            if len(idxs) > 0:
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    color = [int(c) for c in np.random.randint(0, 255, size=(3,))]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            panel.config(image=img)
            panel.image = img
    if is_running:
        panel.after(10, update_camera)

# Função para carregar e processar uma imagem
def process_image():
    global panel, net, classes
    # Abrir uma janela de diálogo para escolher uma imagem
    image_file = filedialog.askopenfilename(initialdir="/", title="Selecione uma imagem",
                                            filetypes=(("Arquivos de Imagem", "*.jpg;*.jpeg;*.png"), ("Todos os arquivos", "*.*")))
    if image_file:
        frame = cv2.imread(image_file)
        if frame is not None:  # Verificar se a imagem foi carregada corretamente
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = resize_with_aspect_ratio(frame, width=640)

            # Detectar objetos usando YOLO
            (H, W) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            ln = net.getLayerNames()
            ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
            layer_outputs = net.forward(ln)

            boxes = []
            confidences = []
            class_ids = []

            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            if len(idxs) > 0:
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    color = [int(c) for c in np.random.randint(0, 255, size=(3,))]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            panel.config(image=img)
            panel.image = img

# Carregar as classes
classes = get_classes(labels_path)

# Carregar a rede YOLO
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Inicializar a interface Tkinter
root = tk.Tk()
root.title("Detecção de Objetos com YOLO")
root.geometry("800x600")

panel = tk.Label(root)
panel.pack(pady=20)

btn_toggle_camera = Button(root, text="Ligar Câmera", command=toggle_camera)
btn_toggle_camera.pack(pady=10)

btn_choose_input = Button(root, text="Selecionar Vídeo", command=lambda: choose_input_mode('video'))
btn_choose_input.pack(pady=10)

btn_choose_image = Button(root, text="Selecionar Imagem", command=process_image)
btn_choose_image.pack(pady=10)

cap = None
is_running = False

root.mainloop()
