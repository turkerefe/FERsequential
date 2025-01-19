import cv2
import numpy as np
from keras.models import load_model

# Modeli yükleme
model = load_model('sequential.h5')

# Kamera akışını başlatma
video = cv2.VideoCapture(0)

# Haar Cascade dosyasını yükleme
faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Sınıf etiketleri
class_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

print("Kamera başlatıldı. 'q' tuşuna basarak çıkabilirsiniz.")

while True:
    ret, frame = video.read()  # Kameradan bir kare okuma
    if not ret:
        print("Kameradan görüntü alınamıyor. Çıkılıyor...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Gri tonlamaya çevirme
    faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)  # Yüz algılama

    for x, y, w, h in faces:
        # Yüz bölgesini çıkarma ve modele uygun hale getirme
        sub_face_img = gray[y:y + h, x:x + w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))

        # Model ile tahmin yapma
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        # Çerçeve ve tahmin etiketlerini görüntüye ekleme
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, class_labels[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Görüntüyü gösterme
    cv2.imshow("Duygu Tanıma", frame)

    # 'q' tuşuna basıldığında çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırakma
video.release()
cv2.destroyAllWindows()
