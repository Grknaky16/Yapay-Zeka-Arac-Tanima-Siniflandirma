import numpy as np
import time
import cv2
import os
import PySimpleGUI as sg
# import PySimpleGUIQt as sg        

y_path = r'yolo-coco'

sg.theme('LightGreen')

gui_confidence = .5     # başlangıç ​​ayarları
gui_threshold = .3      # başlangıç ​​ayarları
camera_number = 0       # 1'den fazla kameranız varsa, hangisinin kullanılacağını seçmek için bu değişkeni değiştirin.

# YOLO modelimizin eğitim aldığı COCO sınıfı etiketlerini yükleyin.
labelsPath = os.path.sep.join([y_path, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# olası her sınıf etiketini temsil etmek için bir renk listesi başlatın.
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# YOLO ağırlıklarına ve model konfigürasyonuna giden yolları türetir.
weightsPath = os.path.sep.join([y_path, "yolov3.weights"])
configPath = os.path.sep.join([y_path, "yolov3.cfg"])

sg.popup_quick_message('Loading YOLO weights from disk.... one moment...', background_color='red', text_color='white')

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# video akışını, çıkış video dosyasının işaretçisini ve çerçeve boyutlarını başlatın.
W, H = None, None
win_started = False
cap = cv2.VideoCapture(camera_number)  # yakalama cihazını başlat
while True:
    # dosyadan veya web kamerasından sonraki kareyi okuyun.
    grabbed, frame = cap.read()

    # Çerçeve yakalanmadıysa yayını durdur.
    if not grabbed:
        break

    # Çerçeve boyutları boşsa, onları yakala.
    if not W or not H:
        (H, W) = frame.shape[:2]

    # Giriş görüntüsünden bir blob oluşturun ve ardından bir iletme gerçekleştirin.
    # Bize sınırlayıcı kutularımızı veren YOLO nesne dedektörünün geçişi ve ilişkili olasılıklar.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # Sırasıyla, algılanan sınırlayıcı kutular, güven ve sınıf kimlikleri listelerimizi başlatın.
    boxes = []
    confidences = []
    classIDs = []

    # Katman çıktılarının her biri üzerinde döngü yapın.
    for output in layerOutputs:
        # Algılamaların her biri üzerinde döngü yapın.
        for detection in output:
            # geçerli nesne algılama
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # zayıf tahminleri tespit edilmesini sağlayarak filtreleme.
            if confidence > gui_confidence:
                # YOLO sınırlayıcı kutunun merkez (x, y) koordinatlarını ve ardından kutuların genişliğini ve yüksekliğini döndürür.
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Sınırlayıcı kutunun üst ve sol köşesini elde etmek için merkez (x, y) koordinatlarını kullanın.
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Sınırlayıcı kutu koordinatları, güven ve sınıf kimlikleri listemizi güncelleyin.
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Zayıf, örtüşen sınırlayıcı kutuları bastırmak için maksimum olmayan bastırma uygulayın.
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, gui_confidence, gui_threshold)

    # En az bir algılamanın mevcut olduğundan emin olun.
    if len(idxs) > 0:
        # Tuttuğumuz dizinler üzerinde döngü yapın.
        for i in idxs.flatten():
            # sınırlayıcı kutu koordinatlarını çıkarın.
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # resmin üzerine bir sınırlayıcı kutu dikdörtgeni çizin ve etiketleyin.
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                       confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    imgbytes = cv2.imencode('.ppm', frame)[1].tobytes()
    # ---------------------------- THE GUI ----------------------------
    if not win_started:
        win_started = True
        layout = [
            [sg.Text('Webcam İle Nesne Takibi', size=(30, 1))],
            [sg.Graph((W, H), (0,0), (W,H), key='-GRAPH-')],
            [sg.Text('Doğruluk'),
             sg.Slider(range=(0, 10), orientation='h', resolution=1, default_value=5, size=(15, 15), key='confidence'),
             sg.Text('Eşik Değeri'),
             sg.Slider(range=(0, 10), orientation='h', resolution=1, default_value=3, size=(15, 15), key='threshold')],
            [sg.Exit()]
        ]
        window = sg.Window('YOLO Webcam', layout, default_element_size=(14, 1), text_justification='right', auto_size_text=False, finalize=True)
        image_elem = window['-GRAPH-']     # type: sg.Graph
    else:
        image_elem.erase()
        image_elem.draw_image(data=imgbytes, location=(0, H))

    event, values = window.read(timeout=0)
    if event is None or event == 'Exit':
        break
    gui_confidence = int(values['confidence']) / 10
    gui_threshold = int(values['threshold']) / 10

print("[INFO] cleaning up...")
window.close()
