# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import PySimpleGUI as sg
sg.theme('light green')
layout = 	[
		[sg.Text('Resimden Nesne Tanıma', size=(20,1), font=('Any',18),text_color='#1c86ee' ,justification='left')],
		[sg.Text('Fotoğraf Adresi'), sg.In(r'Resim Seçin Lütfen',size=(40,1), key='image'), sg.FileBrowse()],
		[sg.Text('Yolo Adresi'), sg.In(r'yolo-coco',size=(40,1), key='yolo'), sg.FolderBrowse()],
		[sg.Text('Doğruluk'), sg.Slider(range=(0,10),orientation='h', resolution=1, default_value=5, size=(15,15), key='confidence')],
		[sg.Text('Eşik Değeri'), sg.Slider(range=(0,10), orientation='h', resolution=1, default_value=3, size=(15,15), key='threshold')],
		[sg.OK(), sg.Cancel(), sg.Stretch()]
			]

window = sg.Window('YOLO', layout,
                   default_element_size=(14,1),
                   text_justification='right',
                   auto_size_text=False)
event, values = window.read()
args = values
window.close()

args['threshold'] = float(args['threshold']/10)
args['confidence'] = float(args['confidence']/10)

labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

#olası her sınıf etiketini temsil etmek için bir renk listesi başlat
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# YOLO ağırlıklarına ve model konfigürasyonuna giden yolları türetme
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Giriş resmini yükleyin ve boyutlarını alın
image = cv2.imread(args["image"])

(H, W) = image.shape[:2]

# YOLO'dan yalnızca ihtiyacımız olan *çıktı* katman adlarını belirleyin.
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# Giriş görüntüsünden bir blob oluşturun ve ardından bir iletme gerçekleştirin.
# Bize sınırlayıcı kutularımızı veren YOLO nesne dedektörünün geçişi ve ilişkili olasılıklar.

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# YOLO'da zamanlama bilgilerini göster
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

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
		if confidence > args["confidence"]:
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
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

# En az bir algılamanın mevcut olduğundan emin olun.
if len(idxs) > 0:
	# Tuttuğumuz dizinler üzerinde döngü yapın.
	for i in idxs.flatten():
		# sınırlayıcı kutu koordinatlarını çıkarın.
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		# resmin üzerine bir sınırlayıcı kutu dikdörtgeni çizin ve etiketleyin.
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

# çıktı görüntüsünü göster
imgbytes = cv2.imencode('.png', image)[1].tobytes()


layout = 	[
		[sg.Text('Yolo Output')],
		[sg.Image(data=imgbytes)],
		[sg.OK(), sg.Cancel()]
			]

window = sg.Window('YOLO',
                   default_element_size=(14,1),
                   text_justification='right',
                   auto_size_text=False).Layout(layout)
event, values = window.Read()
window.Close()

# cv2.imshow("Image", image)
cv2.waitKey(0)