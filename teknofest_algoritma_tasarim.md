# TEKNOFEST 2025 İNSANSIZ DENİZALTI YARIŞMASI
## ALGORİTMA TASARIM SÜRECİ

### 1. GİRİŞ

Bu bölümde, Teknofest 2025 İnsansız Denizaltı Yarışması için geliştirdiğimiz su altı aracının algoritma tasarım süreci detaylı olarak açıklanmaktadır. Yarışma kapsamında aracımızın "Kayıp Hazine Avı: Atlantis'in Peşinde" ve "Su Altı Kabloları Takibi ve Anomali Tespiti" görevlerini otonom olarak gerçekleştirmesi gerekmektedir. Bu doğrultuda kullanılan teknolojiler, algoritmaların geliştirilme süreçleri ve final tasarım kararları bu raporda sunulmaktadır.

### 2. SİSTEM MİMARİSİ

Aracımızın yazılım mimarisi aşağıdaki bileşenlerden oluşmaktadır:

- **Ana İşlemci**: Raspberry Pi 4 (4GB RAM)
- **Yardımcı İşlemci**: ESP32 (Sensör verileri ve motor kontrolü için)
- **Programlama Dili**: Python 3.9
- **Kütüphaneler**:
  - OpenCV (Görüntü işleme)
  - TensorFlow/PyTorch (Yapay zeka modelleri)
  - NumPy (Matematiksel işlemler)
  - pySerial (ESP32 ile seri haberleşme)
  - ROS (Robot Operating System - modüller arası haberleşme)

### 3. GÖREV ANALİZİ VE ALGORİTMA TASARIMI

#### 3.1. Görev 1: Kayıp Hazine Avı: Atlantis'in Peşinde (Otonom)

Bu görevde aracımızın, başlangıç olan karesel alandan varış olan karesel alana tamamen su altından giderek ulaşması ve varış alanında su yüzeyine çıkması gerekmektedir.

#### 3.1.1. Algoritma Yaklaşımı

Görev 1 için "Açık Döngü Navigasyon + Görüntü İşleme ile Doğrulama" yaklaşımını benimsedik. Bu yaklaşımda:

1. **Konum Tespiti**: GPS modülünden alınan konum bilgisi ile başlangıç anında aracın konumu tespit edilir.
2. **Rota Planlaması**: Varış koordinatlarına doğru optimum rota hesaplanır.
3. **Derinlik Kontrolü**: Basınç sensörü ile aracın belirli bir derinlikte sabit kalması sağlanır.
4. **Navigasyon**: IMU (Inertial Measurement Unit) sensörü ile aracın yönelimi kontrol edilir.
5. **Varış Doğrulama**: Kamera ile varış alanı tespit edildiğinde su yüzeyine çıkılır.

#### 3.1.2. Algoritma Akış Diyagramı - Görev 1

```
+-------------------+     +-------------------+     +--------------------+
| Başlangıç Alanı   |     | Konum Bilgisi     |     | Rota Hesaplama     |
| Tespiti           | --> | Elde Etme         | --> | (Varış Alanına)    |
+-------------------+     +-------------------+     +--------------------+
          |
          v
+-------------------+     +-------------------+     +--------------------+
| Su Altına Dalış   |     | Yön ve Derinlik   |     | Sensör Verileri    |
| ve Denge Kontrolü | <-- | Kontrolü          | <-- | Toplama ve İşleme  |
+-------------------+     +-------------------+     +--------------------+
          |
          v
+-------------------+     +-------------------+     +--------------------+
| Varış Alanı       |     | Su Yüzeyine       |     | Görev Tamamlama    |
| Tespiti           | --> | Çıkış            | --> | ve Veri Kaydı      |
+-------------------+     +-------------------+     +--------------------+
```

#### 3.1.3. Konum ve Navigasyon Algoritması - Örnek Kod

```python
import numpy as np
from pyproj import Geod

def calculate_distance_bearing(lat1, lon1, lat2, lon2):
    """Konum koordinatları arasındaki mesafe ve açıyı hesaplar"""
    geod = Geod(ellps="WGS84")
    azimuth, back_azimuth, distance = geod.inv(lon1, lat1, lon2, lat2)
    return distance, azimuth

def navigate_to_target(current_pos, target_pos, depth_target=2.0):
    """Hedef konuma navigasyon algoritması"""
    # Konum bilgilerini ayıkla
    curr_lat, curr_lon = current_pos
    target_lat, target_lon = target_pos
    
    # Mesafe ve açı hesapla
    distance, bearing = calculate_distance_bearing(curr_lat, curr_lon, target_lat, target_lon)
    
    # Hedef mesafeye yaklaşıldıysa (3 metre eşik değeri)
    if distance < 3.0:
        return {
            'thrust': 0.2,  # Yavaşla
            'direction': bearing,
            'depth': depth_target,
            'status': 'approaching_target'
        }
    else:
        return {
            'thrust': 0.6,  # Normal hız
            'direction': bearing,
            'depth': depth_target,
            'status': 'moving_to_target'
        }
```

#### 3.1.4. Derinlik Kontrolü - PID Kontrol Algoritması

Derinlik kontrolünde PID (Proportional-Integral-Derivative) kontrolörü kullanarak aracın su altında sabit derinlikte kalması sağlanmıştır:

```python
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.prev_error = 0
        self.integral = 0
        
    def compute(self, setpoint, measured_value, dt):
        """PID hesaplama fonksiyonu"""
        # Hata hesaplama
        error = setpoint - measured_value
        
        # Integral terimi hesaplama (sınırlı)
        self.integral += error * dt
        self.integral = max(-100, min(100, self.integral))  # Anti-windup
        
        # Türev terimi hesaplama
        derivative = (error - self.prev_error) / dt
        
        # PID çıktısı hesaplama
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        # Bir sonraki hesaplama için hata değerini sakla
        self.prev_error = error
        
        return output

# Kullanım örneği
depth_controller = PIDController(kp=1.2, ki=0.1, kd=0.05)
```

#### 3.1.5. Varış Alanı Tespiti

Varış alanını tespit etmek için şu yaklaşımı kullanıyoruz:

1. Aracın koordinatının, varış alanı merkez koordinatına belirli bir mesafede olduğunu teyit etme
2. Kamera ile görüntü işleme yaparak karesel alanın kenarlarını tespit etme
3. Su yüzeyine çıkış için sinyal üretme

```python
def detect_arrival_area(frame, min_area_size):
    """Kamera görüntüsünden varış alanını tespit etme"""
    # Görüntüyü gri tona çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Gürültü azaltma
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Kenar tespiti
    edges = cv2.Canny(blurred, 50, 150)
    
    # Konturları bul
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # En büyük konturu seç (potansiyel karesel alan)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Minimum alan boyutunu kontrol et
        if area > min_area_size:
            # Dörtgen yaklaşımı
            approx = cv2.approxPolyDP(largest_contour, 0.02*cv2.arcLength(largest_contour, True), True)
            
            # Dörtgen kontrol (4 köşeli)
            if len(approx) == 4:
                return True, approx
            
    return False, None
```

#### 3.2. Görev 2: Su Altı Kabloları Takibi ve Anomali Tespiti (Otonom)

Bu görevde aracımızın, su altındaki kabloyu takip ederek üzerindeki 4 farklı anomali şeklini tespit etmesi ve kaydetmesi gerekmektedir.

#### 3.2.1. Algoritma Yaklaşımı

Görev 2 için "Kablo Takibi + Derin Öğrenme Tabanlı Anomali Tespiti" yaklaşımını benimsedik:

1. **Kablo Takibi**: Görüntü işleme ile kablonun tespit edilmesi ve takibi
2. **Anomali Tespiti**: Önceden eğitilmiş bir CNN (Convolutional Neural Network) modeli ile anomali şekillerinin tanınması
3. **Veri Kaydı**: Tespit edilen anomalilerin isim ve görüntülerinin kaydedilmesi

#### 3.2.2. Algoritma Akış Diyagramı - Görev 2

```
+-------------------+     +-------------------+     +-------------------+
| Kamera            |     | Görüntü Ön        |     | Kablo Hat         |
| Görüntü Alımı     | --> | İşleme            | --> | Tespiti           |
+-------------------+     +-------------------+     +-------------------+
          |
          v
+-------------------+     +-------------------+     +-------------------+
| Kablo Takip       |     | Anomali Bölgesi   |     | Derin Öğrenme ile |
| Algoritması       | --> | Tespiti           | --> | Şekil Tanıma      |
+-------------------+     +-------------------+     +-------------------+
          |
          v
+-------------------+     +-------------------+     +-------------------+
| Anomali İsim      |     | Görüntü ve İsim   |     | Bir Sonraki       |
| Tespiti           | --> | Eşleştirme ve Kayıt| --> | Anomaliye İlerleme|
+-------------------+     +-------------------+     +-------------------+
```

#### 3.2.3. Kablo Tespit ve Takip Algoritması

Kablo tespiti için renk tabanlı segmentasyon ve Hough dönüşümü kullanılmıştır:

```python
def detect_cable(frame):
    """Görüntüde kabloyu tespit etme"""
    # HSV renk uzayına dönüştürme
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Kablonun renk aralığını belirle (örn: siyah kablo)
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([180, 255, 50])
    
    # Maske oluştur
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Gürültü temizleme
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Kenarları tespit et
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    
    # Hough çizgi dönüşümü
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    
    if lines is not None:
        # Çizgileri birleştir ve ortalama yönü hesapla
        x_coords = []
        y_coords = []
        angles = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
            
            # Çizgi açısını hesapla
            if x2 - x1 != 0:  # Sıfıra bölünmeyi önle
                angle = np.arctan((y2 - y1) / (x2 - x1))
                angles.append(angle)
        
        # Kablo merkez noktası
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        
        # Ortalama yönelim açısı
        avg_angle = np.mean(angles) if angles else 0
        
        return True, (center_x, center_y, avg_angle)
    
    return False, None

def follow_cable(frame, prev_center=None):
    """Kabloyu takip etme algoritması"""
    detected, cable_info = detect_cable(frame)
    
    if not detected:
        # Kablo görünmüyorsa, son bilinen konumu kullan
        if prev_center:
            return {
                'status': 'searching',
                'action': 'rotate_and_scan',
                'center': prev_center
            }
        else:
            return {
                'status': 'not_found',
                'action': 'search_pattern',
                'center': None
            }
    
    # Kablo merkezi ve yönelimi
    center_x, center_y, angle = cable_info
    
    # Aracın konumu ile kablo merkezi arasındaki sapma
    frame_center_x = frame.shape[1] // 2
    deviation = center_x - frame_center_x
    
    # Takip komutlarını oluştur
    if abs(deviation) < 30:  # Kabloyu merkezde tut (30 piksel tolerans)
        return {
            'status': 'centered',
            'action': 'move_forward',
            'center': (center_x, center_y),
            'angle': angle
        }
    elif deviation < 0:  # Kablo solda
        return {
            'status': 'tracking',
            'action': 'move_left',
            'center': (center_x, center_y),
            'angle': angle
        }
    else:  # Kablo sağda
        return {
            'status': 'tracking',
            'action': 'move_right',
            'center': (center_x, center_y),
            'angle': angle
        }
```

#### 3.2.4. Anomali Tespiti - CNN Modeli

Anomali tespiti için önceden eğitilmiş bir CNN modeli kullanıyoruz:

```python
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

class AnomalyDetector:
    def __init__(self, model_path):
        """Anomali tespit modeli yükleme"""
        self.model = load_model(model_path)
        self.class_names = ['ALTIGEN', 'DAIRE', 'KARE', 'UCGEN', 'YILDIZ', 
                            'BESGEN', 'DIKDORTGEN', 'ELIPS', 'KALP', 'YAMUK']
        self.input_shape = (224, 224)  # Model giriş boyutu
        
    def preprocess_image(self, image):
        """Görüntüyü model için hazırlama"""
        img = cv2.resize(image, self.input_shape)
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Batch boyutu ekleme
        return img
        
    def detect(self, image, confidence_threshold=0.7):
        """Görüntüdeki anomaliyi tespit etme"""
        processed_img = self.preprocess_image(image)
        predictions = self.model.predict(processed_img)[0]
        
        # En yüksek olasılık
        max_prob_idx = np.argmax(predictions)
        max_prob = predictions[max_prob_idx]
        
        if max_prob >= confidence_threshold:
            return True, self.class_names[max_prob_idx], max_prob
        else:
            return False, None, max_prob
            
    def save_detection(self, image, class_name, timestamp):
        """Tespit edilen anomaliyi kaydetme"""
        filename = f"anomaly_{class_name}_{timestamp}.jpg"
        cv2.imwrite(f"detections/{filename}", image)
        
        # Tespit logunu kaydet
        with open("detections/anomaly_log.txt", "a") as f:
            f.write(f"{timestamp},{filename},{class_name}\n")
            
        return filename
```

#### 3.2.5. Anomali Bölgesi Tespiti

Kablo üzerindeki anomali bölgelerini tespit etmek için özel bir algoritma:

```python
def detect_anomaly_region(frame, cable_info):
    """Kablo üzerinde potansiyel anomali bölgelerini tespit etme"""
    if not cable_info:
        return False, None
    
    center_x, center_y, angle = cable_info
    
    # Kablonun etrafındaki ROI (Region of Interest) belirleme
    roi_width = 300
    roi_height = 300
    
    # ROI koordinatları (görüntü sınırlarını aşmamasını sağla)
    x1 = max(0, center_x - roi_width//2)
    y1 = max(0, center_y - roi_height//2)
    x2 = min(frame.shape[1], center_x + roi_width//2)
    y2 = min(frame.shape[0], center_y + roi_height//2)
    
    # ROI'yi kes
    roi = frame[y1:y2, x1:x2]
    
    if roi.size == 0:  # Boş ROI kontrolü
        return False, None
        
    # Görüntü işleme ile şekil tespiti
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    
    # Konturları bul
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Belirli bir boyuttan büyük konturları seç (potansiyel anomaliler)
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Minimum alan eşiği
        if area > 500:
            # Kontur çevreleyen dikdörtgen
            x, y, w, h = cv2.boundingRect(contour)
            
            # Global koordinatlara dönüştür
            global_x1 = x1 + x
            global_y1 = y1 + y
            global_x2 = x1 + x + w
            global_y2 = y1 + y + h
            
            anomaly_region = frame[global_y1:global_y2, global_x1:global_x2]
            
            if anomaly_region.size > 0:
                return True, anomaly_region
    
    return False, None
```

### 4. ENTEGRASYON VE HABERLEŞME

Raspberry Pi ve ESP32 arasındaki haberleşme için seri iletişim protokolü kullanılmıştır:

```python
import serial
import json
import time

class CommunicationManager:
    def __init__(self, port='/dev/ttyUSB0', baud_rate=115200):
        """ESP32 ile haberleşme yöneticisi"""
        self.serial_conn = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)  # Bağlantı kurulması için bekle
        
    def send_command(self, command_dict):
        """ESP32'ye komut gönderme"""
        command_json = json.dumps(command_dict)
        self.serial_conn.write(f"{command_json}\n".encode())
        
    def read_sensor_data(self):
        """ESP32'den sensör verilerini okuma"""
        if self.serial_conn.in_waiting > 0:
            data = self.serial_conn.readline().decode().strip()
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return None
        return None
        
    def close(self):
        """Bağlantıyı kapatma"""
        if self.serial_conn.is_open:
            self.serial_conn.close()
```

### 5. FİNAL TASARIM SEÇİMİ VE GEREKÇELERİ

Geliştirme sürecinde çeşitli alternatifler arasından aşağıdaki tasarım kararları alınmıştır:

#### 5.1. Görev 1 - Navigasyon Algoritması

**Seçilen Tasarım**: Açık döngü navigasyon + görüntü işleme ile doğrulama

**Alternatifler ve Değerlendirme**:
1. **SLAM (Simultaneous Localization and Mapping)**: Su altında sonar ve kamera tabanlı SLAM algoritmalarını test ettik ancak donanım gereksinimlerinin yüksek olması ve gerçek zamanlı performans sorunları nedeniyle tercih edilmedi.
2. **Tamamen konum tabanlı navigasyon**: Sadece GPS ve IMU sensörlerini kullanarak navigasyon planlamak hata payını arttırdığından, görüntü işleme ile doğrulama ekleyerek hibrit bir çözüm oluşturduk.

**Gerekçe**: Seçilen hibrit yaklaşım, her iki yöntemin avantajlarını birleştirerek hem hızlı navigasyon hem de doğru varış alanı tespiti sağlamaktadır. Ayrıca Raspberry Pi 4'ün işlemci gücünü optimum kullanarak gerçek zamanlı performans göstermektedir.

#### 5.2. Görev 2 - Anomali Tespit Algoritması

**Seçilen Tasarım**: Derin öğrenme tabanlı CNN modeli

**Alternatifler ve Değerlendirme**:
1. **Geleneksel Görüntü İşleme**: Şablon eşleme ve kontur analizi yöntemleri test edildi ancak su altı ortamındaki ışık ve bulanıklık değişimlerine karşı yeterince dayanıklı olmadığı görüldü.
2. **YOLO (You Only Look Once)**: Gerçek zamanlı nesne tespit algoritmalarını değerlendirdik ancak eğitim süreci ve hesaplama maliyeti daha yüksekti.

**Gerekçe**: CNN tabanlı sınıflandırma modeli, su altı görüntülerindeki zorlayıcı koşullarda bile (düşük ışık, bulanıklık) yüksek başarı oranı göstermiştir. Transfer öğrenme kullanarak kısıtlı veri setiyle bile iyi sonuçlar elde edilebildi ve Raspberry Pi üzerinde kabul edilebilir bir hızda çalışmaktadır.

### 6. TEST VE DOĞRULAMA

Geliştirilen algoritmaların performansını aşağıdaki kriterlere göre test ettik:

1. **Doğruluk**: Algoritmaların farklı koşullar altında doğru sonuçlar üretme yeteneği
2. **Hız**: Gerçek zamanlı performans gereksinimleri
3. **Güvenilirlik**: Farklı çevre koşullarında kararlı çalışabilme
4. **Enerji Verimliliği**: Batarya ömrünün optimizasyonu

Test sonuçlarımız, seçilen algoritmaların yarışma gereksinimlerini karşıladığını ve 10 dakikalık görev süresinde tamamlanabileceğini göstermiştir.

### 7. SONUÇ VE GELECEK ÇALIŞMALAR

Teknofest 2025 İnsansız Denizaltı Yarışması için geliştirdiğimiz algoritma tasarım süreci, otonom su altı araçlarının zorlu görevleri başarıyla tamamlayabilmesi için optimize edilmiştir. Raspberry Pi 4 ve ESP32 tabanlı mimari ile Python programlama dilinin sağladığı esneklik, hızlı prototipleme ve geliştirmeyi mümkün kılmıştır.

Gelecek çalışmalarda, özellikle yapay zeka modellerinin optimize edilmesi, daha karmaşık ortamlarda test edilmesi ve enerji verimliliğinin artırılması planlanmaktadır.

### KAYNAKLAR

1. OpenCV Belgeleri: https://docs.opencv.org/
2. TensorFlow Lite Belgeleri: https://www.tensorflow.org/lite
3. Raspberry Pi Belgeleri: https://www.raspberrypi.org/documentation/
4. ROS Belgeleri: http://wiki.ros.org/
5. ESP32 Belgeleri: https://docs.espressif.com/projects/esp-idf/