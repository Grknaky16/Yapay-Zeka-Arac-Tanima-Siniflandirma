# import the necessary packages
import numpy as np
# import argparse
import imutils
import time
import cv2
import os
import PySimpleGUI as sg

i_vid = r'Video Seçin Lütfen'
y_path = r'yolo-coco'
sg.theme('light green')

layout = 	[
		[sg.Text('Videodan Nesne Tanıma', size=(20,1), font=('Any',18),text_color='#1c86ee' ,justification='left')],
		[sg.Text('Video Adresi'), sg.In(i_vid,size=(40,1), key='input'), sg.FileBrowse()],
		# [sg.Text('Path to output video'), sg.In(o_vid,size=(40,1), key='output'), sg.FileSaveAs()],
		[sg.Text('Yolo Adresi'), sg.In(y_path,size=(40,1), key='yolo'), sg.FolderBrowse()],
		[sg.Text('Doğruluk'), sg.Slider(range=(0,1),orientation='h', resolution=.1, default_value=.5, size=(15,15), key='confidence')],
		[sg.Text('Eşik Değeri'), sg.Slider(range=(0,1), orientation='h', resolution=.1, default_value=.3, size=(15,15), key='threshold')],
		[sg.OK(), sg.Cancel()]
			]

window = sg.Window('YOLO Video', layout,
                   default_element_size=(14,1),
                   text_justification='right',
                   auto_size_text=False)
event, values = window.read()
if event is None or event =='Cancel':
	exit()
args = values

window.close()


# imgbytes = cv2.imencode('.png', image)[1].tobytes()

# YOLO modelimizin eğitim aldığı COCO sınıfı etiketlerini yükleyin.
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# olası her sınıf etiketini temsil etmek için bir renk listesi başlat
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# YOLO ağırlıklarına ve model konfigürasyonuna giden yolları türetme
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# Giriş videosu yükleyin ve boyutlarını alın
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# video dosyasındaki toplam kare sayısını belirlemeye çalışın.
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# video dosyasındaki toplam kare sayısı belirlenmeye çalışılırken bir hata oluşursa.
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# video dosyası akışından kareler üzerinde döngü yapın.
win_started = False
while True:
	# dosyadan sonraki kareyi okuma.
	(grabbed, frame) = vs.read()

	# Çerçeve yakalanmadıysa, akışın sonuna ulaşır.
	if not grabbed:
		break

	# Çerçeve boyutları boşsa, onları yakala.
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# Giriş görüntüsünden bir blob oluşturun ve ardından bir iletme gerçekleştirin.
	# sınırlayıcı kutularımızı veren YOLO nesne dedektörünün geçişi ve ilişkili olasılıklar.
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
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	# çıktı görüntüsünü göster
	imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto

	if not win_started:
		win_started = True
		layout = [
			[sg.Text('Yolo Output')],
			[sg.Image(data=imgbytes, key='_IMAGE_')],
			[sg.Exit()]
		]
		window = sg.Window('YOLO Output', layout,
                           default_element_size=(14, 1),
                           text_justification='right',
                           auto_size_text=False, finalize=True)
		image_elem = window['_IMAGE_']
	else:
		image_elem.update(data=imgbytes)

	event, values = window.read(timeout=0)
	if event is None or event == 'Exit':
		break


window.close()

# dosya işaretçilerini serbest bırakın.
print("[INFO] cleaning up...")
writer.release()
vs.release()