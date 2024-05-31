# Detecção de Objetos com YOLO e Tkinter

Este projeto utiliza a rede YOLO (You Only Look Once) para detecção de objetos em tempo real através de webcam, vídeos ou imagens. A interface gráfica é construída com Tkinter.

## Funcionalidades

- **Ligar/Desligar Câmera:** Permite iniciar e parar a captura de vídeo da webcam.
- **Selecionar Vídeo:** Permite selecionar um arquivo de vídeo para detecção de objetos.
- **Selecionar Imagem:** Permite selecionar uma imagem para detecção de objetos.

## Requisitos

- Python 3.x
- OpenCV
- NumPy
- Tkinter
- Pillow

## Instalação

1. Clone o repositório:
    ```bash
    git clone https://github.com/vithorcamara/DetectorDeObjetos.git
    cd DetectorDeObjetos
    ```

2. Instale as dependências:
    ```bash
    pip install opencv-python-headless numpy pillow
    ```

3. Baixe os arquivos do YOLO:
    - [YOLOv3 Config](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
    - [YOLOv3 Weights](https://pjreddie.com/media/files/yolov3.weights)
    - [COCO Names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

    Coloque os arquivos `yolov3.cfg`, `yolov3.weights` e `coco.names` no diretório do projeto.

## Como Usar

1. Execute o script principal:
    ```bash
    python main.py
    ```

2. Na interface gráfica, você pode:
    - **Ligar/Desligar a Câmera** para capturar vídeo em tempo real da webcam.
    - **Selecionar Vídeo** para escolher um arquivo de vídeo e detectar objetos.
    - **Selecionar Imagem** para escolher uma imagem e detectar objetos.

## Estrutura do Projeto

```plaintext
.
├── yolov3.cfg
├── yolov3.weights
├── coco.names
├── main.py
└── README.md
