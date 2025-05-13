# Yazılım Tasarım Süreci

## 1. Yazılım Mimarisi ve Programlama Dili Seçimi

### 1.1. Programlama Dili Tercihi: Python

Aracımızın tüm yazılım bileşenleri Python programlama dili kullanılarak geliştirilmiştir. Python'un tercih edilme sebepleri:

- **Hızlı Geliştirme**: Prototip oluşturma ve geliştirme sürecini hızlandıran yüksek seviyeli yapısı
- **Zengin Kütüphane Ekosistemi**: OpenCV, NumPy, Yolo gibi görüntü işleme ve yapay zeka için güçlü kütüphanelerin varlığı
- **Platform Bağımsızlığı**: Raspberry Pi ve ESP32 gibi farklı platformlarda çalışabilmesi
- **Topluluk Desteği**: Yaygın kullanımı sayesinde karşılaşılan sorunların çözümüne kolay erişim
- **Düşük Donanım Gereksinimleri**: Sınırlı kaynaklara sahip gömülü sistemlerde bile verimli çalışabilmesi

### 1.2. Yazılım Bileşenleri ve Yapısı

```
proje/
│
├── main.py                  # Ana program
├── config/                  # Konfigürasyon dosyaları
│   ├── settings.py
│   └── mission_params.py
│
├── sensors/                 # Sensör modülleri
│   ├── pressure.py
│   ├── imu.py
│   └── camera.py
│
├── navigation/              # Navigasyon modülleri
│   ├── path_planning.py
│   ├── localization.py
│   └── pid_controller.py
│
├── vision/                  # Görüntü işleme modülleri
│   ├── cable_detection.py
│   ├── anomaly_detection.py
│   └── object_recognition.py
│
├── motor/                   # Motor kontrol modülleri
│   ├── thruster_control.py
│   └── stabilization.py
│
├── ai/                      # Yapay zeka modülleri
│   ├── ml_models/
│   └── shape_classifier.py
│
└── utils/                   # Yardımcı fonksiyonlar
    ├── data_logger.py
    └── communication.py
```

## 2. Donanım-Yazılım Entegrasyonu

### 2.1. Raspberry Pi 4 ve ESP32 İş Bölümü

**Raspberry Pi 4 (Ana İşlemci):**
- Yüksek seviyeli algoritmalar
- Görüntü işleme ve nesne tanıma
- Navigasyon ve yörünge planlama
- Görev yönetimi ve karar verme
- Veri kayıt ve loglama

**ESP32 (Yardımcı İşlemci):**
- Motor kontrolü ve PWM sinyali üretimi
- Sensor veri toplama ve ön işleme
- Düşük seviyeli iletişim protokolleri
- Gerçek zamanlı tepki gerektiren işlemler
- Güç yönetimi

### 2.2. İletişim Protokolleri

- **Raspberry Pi 4 ve ESP32 arasında:** UART/Serial iletişim (115200 baud rate)
- **Raspberry Pi 4 ve Sensörler arasında:** I2C ve SPI protokolleri
- **ESP32 ve Motor Sürücüleri arasında:** PWM sinyalleri

## 3. Görev 1: Kayıp Hazine Avı Yazılım Uygulaması

## 3. Görev 1: Kayıp Hazine Avı Yazılım Uygulaması

### 3.1. Engel Algılama ve Otonom Hareket Yazılımı
```python
# Engel algılama ve otonom hareket örneği
def autonomous_obstacle_avoidance():
    # Sensörlerden engel verilerini oku
    front_distance = read_distance_sensor(SENSOR_FRONT)
    left_distance = read_distance_sensor(SENSOR_LEFT)
    right_distance = read_distance_sensor(SENSOR_RIGHT)
    
    # Güvenli mesafe eşik değeri
    safe_distance = 20  # cm
    
    # Engellere göre hareketi belirle
    if front_distance < safe_distance:
        # Önde engel varsa
        if left_distance > right_distance:
            # Sol taraf daha açıksa sola dön
            set_motors(0, -turn_speed)  # Sola dönüş
        else:
            # Sağ taraf daha açıksa sağa dön
            set_motors(0, turn_speed)  # Sağa dönüş
    else:
        # Yol açıksa ilerle, ancak yanlardan da engelleri kontrol et
        forward_speed = base_speed
        turn_adjustment = 0
        
        # Yan sensörlere göre hafif yön düzeltmeleri
        if left_distance < safe_distance:
            turn_adjustment = turn_speed * 0.5  # Sağa doğru hafif düzeltme
        elif right_distance < safe_distance:
            turn_adjustment = -turn_speed * 0.5  # Sola doğru hafif düzeltme
            
        # Motor komutlarını güncelle
        set_motors(forward_speed, turn_adjustment)
    
    return check_target_reached()  # Hedef kontrolü
```

<img width="243" alt="Ekran Resmi 2025-05-13 10 43 58" src="https://github.com/user-attachments/assets/f0d53446-16f4-4e5e-a578-43caf3e3193b" />
Görsel: Engel tanıtımı ve kablo takip çalışmaları

### 3.2. Derinlik Kontrolü için PID Yazılımı

PID (Proportional-Integral-Derivative) kontrolcüsü, aracın sabit derinlikte kalmasını veya hedef derinliğe ulaşmasını sağlar:

```python
class PIDController:
    def __init__(self, kp, ki, kd, output_limits=None):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.output_limits = output_limits
        
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()
    
    def compute(self, error):
        # Zaman farkını hesapla
        current_time = time.time()
        dt = current_time - self.last_time
        
        # PID bileşenlerini hesapla
        p_term = self.kp * error
        self.integral += error * dt
        i_term = self.ki * self.integral
        d_term = self.kd * (error - self.prev_error) / dt if dt > 0 else 0
        
        # Çıktıyı hesapla
        output = p_term + i_term + d_term
        
        # Çıktıyı sınırla (eğer sınırlar belirtilmişse)
        if self.output_limits:
            output = max(min(output, self.output_limits[1]), self.output_limits[0])
        
        # Bir sonraki hesaplama için değerleri sakla
        self.prev_error = error
        self.last_time = current_time
        
        return output
```

## 4. Görev 2: Kablo Takibi ve Anomali Tespiti Yazılım Uygulaması

### 4.1. Kablo Takibi Yazılım Uygulaması

OpenCV kütüphanesi kullanılarak geliştirilen kablo takip yazılımı:

```python
def detect_cable(frame):
    # Görüntüyü gri tonlamaya çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Gürültü azaltma için Gaussian bulanıklaştırma uygula
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Kenar tespiti için Canny algoritması kullanımı
    edges = cv2.Canny(blurred, 50, 150)
    
    # Kenarları daha belirgin hale getirmek için genişletme işlemi
    dilated = cv2.dilate(edges, None, iterations=1)
    
    # Hough çizgi dönüşümü ile çizgileri tespit et
    lines = cv2.HoughLinesP(dilated, 1, np.pi/180, threshold=50, 
                           minLineLength=50, maxLineGap=20)
    
    if lines is not None:
        # Tespit edilen çizgileri işle ve kablo yönünü belirle
        cable_direction = process_detected_lines(lines, frame.shape)
        return True, cable_direction
    else:
        return False, None
```

<img width="382" alt="Ekran Resmi 2025-05-13 10 40 33" src="https://github.com/user-attachments/assets/4fe70d92-097d-4207-aed0-d8948205a71f" />
Görsel: Kablo takip çalışmaları

### 4.2. Anomali Tespiti Yazılım Uygulaması

Su altı aracının çevresindeki geometrik cisimleri tespit edebilmesi amacıyla, gerçek zamanlı görüntü işleme ve derin öğrenme yöntemleri birlikte kullanılmıştır. Cisimlerin konum ve şekil bilgileri için YOLOv5 tabanlı bir nesne tanıma modeli eğitilmiştir. Tespit edilen nesnelerin renk bilgisi, YOLO tarafından belirlenen sınırlayıcı kutular (bounding box) içerisinden alınan piksel verileri OpenCV kütüphanesi ile analiz edilerek HSV renk uzayında hesaplanmıştır. Böylece her bir nesnenin şekli, rengi ve sayısı anlık olarak belirlenebilmiştir.

<img width="967" alt="Ekran Resmi 2025-05-13 10 37 39" src="https://github.com/user-attachments/assets/efcd22c9-b248-405e-b77d-571ea2d348fe" />
Görsel: Şeklin tanınması çalışmaları

```python
import cv2
import torch
import numpy as np
from datetime import datetime
import os

# Klasörleri oluştur (varsa sorun olmaz)
os.makedirs("anomalies", exist_ok=True)

# YOLOv5 modelini yükle (önceden eğitilmiş modelini path'e göre koy)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt')
model.conf = 0.5  # minimum güven skoru

# Sınıf isimlerini al
class_names = model.names  # örnek: ['daire', 'üçgen', 'kare']

# Renk tespit fonksiyonu (HSV uzayında)
def get_dominant_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg = cv2.mean(hsv)[:3]
    h, s, v = avg

    if h < 10 or h > 160:
        return "Kırmızı"
    elif 10 < h < 35:
        return "Sarı"
    elif 35 < h < 85:
        return "Yeşil"
    elif 85 < h < 130:
        return "Mavi"
    else:
        return "Bilinmeyen"

# Tespit ve görselleştirme fonksiyonu
def detect_shapes_and_colors(frame):
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    for *box, conf, cls_id in detections:
        x1, y1, x2, y2 = map(int, box)
        label = class_names[int(cls_id)]

        # Nesnenin alanını al
        roi = frame[y1:y2, x1:x2]
        color = get_dominant_color(roi)

        # Kaydet
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"anomalies/{timestamp}_{label}_{color}.jpg"
        cv2.imwrite(filename, roi)

        # Ekrana çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        text = f"{label}, {color}"
        cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    return frame

# Kamera aç (0: dahili, 1: harici kamera olabilir)
cap = cv2.VideoCapture(0)

print("Başlatıldı. Çıkmak için 'q' tuşuna bas.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    # Cisimleri tespit et
    processed_frame = detect_shapes_and_colors(frame)

    # Göster
    cv2.imshow("Canlı Tespit", processed_frame)

    # Çıkış tuşu
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kapat
cap.release()
cv2.destroyAllWindows()

```

## 5. Model Eğitimi ve Veri Hazırlama

### 5.1. Konvolüsyonel Sinir Ağı (CNN) Eğitimi

10 farklı şekli (anomaliyi) tanımak için eğitilen CNN modeli:

```python
def create_model():
    # ResNet18 ön eğitimli modelini temel al (transfer öğrenme)
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Transfer öğrenme için ana model katmanlarını dondur
    for layer in base_model.layers:
        layer.trainable = False
    
    # Sınıflandırma katmanları ekle
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')  # 10 farklı şekil sınıfı
    ])
    
    # Modeli derle
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

### 5.2. Veri Artırma Teknikleri

Su altı koşullarında daha dayanıklı model eğitimi için veri artırma teknikleri:

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomFlip("horizontal")
])
```

## 6. Performans Optimizasyonu Teknikleri

### 6.1. Görüntü İşleme Optimizasyonu

```python
def optimize_image_processing(frame):
    # Görüntü boyutunu küçült
    resized = cv2.resize(frame, (320, 240))
    
    # ROI (İlgi Bölgesi) belirle ve sadece o bölgeyi işle
    roi = resized[40:200, 0:320]
    
    # Dönüş yaparken daha düşük çözünürlük kullan
    if is_turning:
        roi = cv2.resize(roi, (160, 120))
    
    return roi
```

### 6.2. Model Optimizasyonu ve Hafif Modeller

TensorFlow Lite dönüşümü ile modellerin Raspberry Pi'de daha verimli çalışması sağlanır:

```python
def convert_to_tflite(model_path, output_path):
    # Keras modelini yükle
    model = tf.keras.models.load_model(model_path)
    
    # TensorFlow Lite dönüştürücüsü
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimizasyonları etkinleştir
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Dönüşümü gerçekleştir
    tflite_model = converter.convert()
    
    # Modeli kaydet
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
```

## 7. Çoklu İş Parçacığı (Threading) Kullanımı

Raspberry Pi'nin çok çekirdekli yapısından yararlanmak ve gerçek zamanlı performansı artırmak için çoklu iş parçacığı kullanımı:

```python
import threading

# Kamera görüntü işleme iş parçacığı
def camera_thread_function():
    while not stop_flag:
        ret, frame = camera.read()
        if ret:
            processed_frame = process_frame(frame)
            with frame_lock:
                current_frame = processed_frame
        time.sleep(0.05)

# Kontrol iş parçacığı
def control_thread_function():
    while not stop_flag:
        with frame_lock:
            if current_frame is not None:
                # Kablo takibi veya anomali tespiti yap
                process_and_control(current_frame)
        time.sleep(0.1)

# İş parçacıklarını başlat
frame_lock = threading.Lock()
current_frame = None
stop_flag = False

camera_thread = threading.Thread(target=camera_thread_function)
control_thread = threading.Thread(target=control_thread_function)

camera_thread.start()
control_thread.start()
```

## 8. Hata Yönetimi ve Güvenlik Mekanizmaları

Deniz altı ortamında oluşabilecek beklenmedik durumlara karşı geliştirilen güvenlik yazılımları:

```python
def safety_monitor():
    while True:
        # Batarya seviyesini kontrol et
        battery_level = get_battery_level()
        if battery_level < BATTERY_THRESHOLD:
            trigger_emergency_surface()
        
        # Su sızıntısı sensörlerini kontrol et
        if is_leak_detected():
            trigger_emergency_surface()
            shutdown_electronics()
        
        # İletişim kesilmesini kontrol et
        if communication_timeout():
            execute_return_home_procedure()
        
        time.sleep(1)
```

## 9. Sonuç ve Gelecek Geliştirmeler

Yazılım sistemimiz, Teknofest 2025 görevlerini başarıyla tamamlamak için gerekli tüm bileşenleri içermektedir. Python programlama dilinin sağladığı avantajlar ve zengin kütüphane ekosistemi sayesinde, kısıtlı donanım kaynaklarına rağmen verimli bir şekilde çalışan çözümler geliştirilmiştir.

Gelecekte yapılması planlanan yazılım geliştirmeleri:

- Derin öğrenme modeli PyTorch tabanlı YOLOv5 ile çalıştırılmış, gerçek zamanlı nesne ve renk tespiti için OpenCV ile entegre edilmiştir. 
- ROS (Robot Operating System) entegrasyonu
- Daha gelişmiş sensör füzyon algoritmalarının uygulanması
- Edge TPU gibi yapay zeka hızlandırıcıların entegrasyonu
