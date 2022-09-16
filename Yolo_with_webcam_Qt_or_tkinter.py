# import the necessary packages
import numpy as np
# import argparse
import imutils
import time
import cv2
import os
# import PySimpleGUIQt as sg
import PySimpleGUI as sg

i_vid = r'Resim & Video Seçiniz'
o_vid = r'Kayıt Yolunu Seçin'
y_path = r'yolo-coco'
sg.theme('LightGreen')
layout = 	[
		[sg.Text(' Nesne Tanıma', size=(22,1), font=('Any',18),text_color='#1c86ee' ,justification='left')],
		[sg.Text('Resim & Video Adresi'), sg.In(i_vid,size=(40,1), key='input'), sg.FileBrowse()],
		[sg.Text('Video Kayıt Adresi'), sg.In(o_vid,size=(40,1), key='output'), sg.FileSaveAs()],
		[sg.Text('Yolo Adresi'), sg.In(y_path,size=(40,1), key='yolo'), sg.FolderBrowse()],
		[sg.Text('Doğruluk'), sg.Slider(range=(0,10),orientation='h', resolution=1, default_value=5, size=(15,15), key='confidence'), sg.T('  ', key='_CONF_OUT_')],
		[sg.Text('Eşik Değeri'), sg.Slider(range=(0,10), orientation='h', resolution=1, default_value=3, size=(15,15), key='threshold'), sg.T('  ', key='_THRESH_OUT_')],
		[sg.Text(' '*8), sg.Checkbox('Webcam Kullanmak İçin', key='_WEBCAM_')],
		[sg.Text(' '*8), sg.Checkbox('Diske Yaz', key='_DISK_')],
		[sg.OK(), sg.Cancel(), sg.Stretch()],
			]

win = sg.Window('YOLO Webcam', layout,
				default_element_size=(21,1),
				text_justification='right',
				auto_size_text=False)
event, values = win.read()
if event is None or event =='Cancel':
	exit()
write_to_disk = values['_DISK_']
use_webcam = values['_WEBCAM_']
args = values

win.Close()

# imgbytes = cv2.imencode('.png', image)[1].tobytes()

gui_confidence = args["confidence"]/10
gui_threshold = args["threshold"]/10

# YOLO modelimizin eğitim aldığı COCO sınıfı etiketlerini yükleyin.
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# olası her sınıf etiketini temsil etmek için bir renk listesi.
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# YOLO ağırlıklarına ve model konfigürasyonuna giden yolları türetir.
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# video akışını, çıkış video dosyasının işaretçisini ve çerçeve boyutlarını başlatın.
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# video dosyasındaki toplam kare sayısını belirlemeye çalışın.
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# video dosyasındaki toplam kare sayısı belirlenmeye çalışılırken bir hata mesajı.
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# video dosyası akışından kareler üzerinde döngü yapın.
win_started = False
if use_webcam:
	cap = cv2.VideoCapture(0)
while True:
	# dosyadan veya web kamerasından sonraki kareyi okuyun.
	if use_webcam:
		grabbed, frame = cap.read()
	else:
		grabbed, frame = vs.read()

	# Çerçeve yakalanmadıysa yayını durdur.
	if not grabbed:
		break

	# Çerçeve boyutları boşsa, onları yakala.
	if W is None or H is None:
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

				# Sınırlayıcı kutu koordinatları, doğruluk ve sınıf kimlikleri listemizi güncelleyin.
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
	if write_to_disk:
		if writer is None:
			# video yazıcısını başlatın.
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(frame.shape[1], frame.shape[0]), True)

			# tek karenin işlenmesi hakkında bazı bilgiler.
			if total > 0:
				elap = (end - start)
				print("[INFO] single frame took {:.4f} seconds".format(elap))
				print("[INFO] estimated total time to finish: {:.4f}".format(
					elap * total))

		# çıktı çerçevesini diske yaz
		writer.write(frame)
	imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto

	if not win_started:
		win_started = True
		layout = [
			[sg.Text('Yolo Playback in PySimpleGUI Window', size=(30,1))],
			[sg.Image(data=imgbytes, key='_IMAGE_')],
			[sg.Text('Confidence'),
			 sg.Slider(range=(0, 10), orientation='h', resolution=1, default_value=5, size=(15, 15), key='confidence'),
			sg.Text('Threshold'),
			 sg.Slider(range=(0, 10), orientation='h', resolution=1, default_value=3, size=(15, 15), key='threshold')],
			[sg.Exit()]
		]
		win = sg.Window('YOLO Output', layout,
						default_element_size=(14, 1),
						text_justification='right',
						auto_size_text=False, finalize=True)
		image_elem = win['_IMAGE_']
	else:
		image_elem.Update(data=imgbytes)

	event, values = win.read(timeout=0)
	if event is None or event == 'Exit':
		break
	gui_confidence = values['confidence']/10
	gui_threshold = values['threshold']/10
    
event, values = win.read()
args = values
win.close()


# dosya işaretçilerini serbest bırakın.
print("[INFO] cleaning up...")
writer.release() if writer is not None else None
vs.release()