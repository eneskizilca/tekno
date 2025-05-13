# Algoritma Tasarım Süreci

## 1. Giriş ve Genel Bakış

İnsansız denizaltı aracımız, Teknofest 2025 yarışma görevlerini başarıyla tamamlamak için, otonomluk ön planda tutularak tasarlanmıştır. Araç, belirlenen görevleri yerine getirmek için aşağıdaki temel algoritma bileşenlerini içermektedir:

- Sensör Veri İşleme ve Füzyon Algoritmaları
- Navigasyon ve Rota Planlama Algoritmaları
- Görüntü İşleme ve Nesne Tanıma Algoritmaları
- Yapay Zeka Tabanlı Karar Verme Mekanizmaları
- Motor Kontrol ve Stabilizasyon Algoritmaları

Bu bileşenler, Raspberry Pi 4 ana işlemci ve ESP32 yardımcı kontrol biriminde çalışacak şekilde geliştirilmiştir. Tüm algoritmalar Python programlama dili kullanılarak kodlanmıştır.

## 2. Görev Analizi ve Algoritma Gereksinimleri

### 2.1. Görev 1: Kayıp Hazine Avı - Atlantis'in Peşinde

Bu görev için gerekli algoritma yetenekleri:
- GPS/Koordinat tabanlı navigasyon
- Su altında otonom hareket ve derinlik kontrolü
- Engel tespiti ve kaçınma
- Hedef alanı tanıma ve doğrulama
- Su üstüne kontrollü çıkış

### 2.2. Görev 2: Su Altı Kabloları Takibi ve Anomali Tespiti

Bu görev için gerekli algoritma yetenekleri:
- Kablo hattı tespiti ve takibi
- Görüntü işleme tabanlı anomali tespiti
- Şekil tanıma ve sınıflandırma
- Veri kaydetme ve raporlama

## 3. Sistem Mimarisi ve Algoritma Entegrasyonu

Algoritmalarımız aşağıdaki ana bileşenlerden oluşmaktadır:

![Sistem Mimarisi](https://placeholder/system_mimarisi)

### 3.1. Donanım-Yazılım Etkileşimi

- **Raspberry Pi 4**: Ana işlemci, yüksek seviye algoritmalar, görüntü işleme, yapay zeka ve karar verme
- **ESP32**: Düşük seviye kontrol, sensör veri toplama, motor sürücü kontrolü

### 3.2. Yazılım Katmanları

1. **Sensör Katmanı**: Ham veri toplama ve ön işleme
2. **Algılama Katmanı**: Çevre algılama ve durumsal farkındalık
3. **Karar Verme Katmanı**: Görev planlaması ve yörünge hesaplama
4. **Kontrol Katmanı**: Motor ve aktuatör kontrolü

## 4. Algoritmalar ve Akış Diyagramları

### 4.1. Ana Sistem Algoritması

Aracın genel çalışma prensibini gösteren ana sistem algoritması akış diyagramı:

```
[Başlangıç] → [Sistem Başlatma] → [Görev Seçimi] → [Görev 1 veya Görev 2] → [Görev Tamamlandı mı?] → [Sonlandırma]
```

![Ana Algoritma Akış Diyagramı](https://placeholder/ana_algoritma)

### 4.2. Görev 1: Kayıp Hazine Avı Algoritmasi

```
[Başlangıç] → [Koordinat Bilgilerini Al] → [Başlangıç Alanına Git] → [Su Altına Dal] → [Hedef Koordinatlara Doğru İlerle] → [Konum Kontrol] → [Varış Alanında mıyız?] → [Evet ise Su Yüzeyine Çık] → [Görev Tamamlandı]
```

![Görev 1 Algoritma Akış Diyagramı](https://placeholder/gorev1_akis)

#### 4.2.1. Koordinat Tabanlı Navigasyon Algoritması

Aracımızın koordinat bilgilerini kullanarak rotasını belirlemesi ve hedefe yönelmesi için geliştirilen algoritma:

```
1. GPS üzerinden başlangıç ve hedef koordinatlarını al
2. Hedef vektörü hesapla (yön ve mesafe)
3. Su altında basınç sensörü ve IMU ile konum takibi yap
4. Ultrasonik sensörler ile engel tespiti
5. Rota düzeltmesi gerekiyorsa yeni yörünge hesapla
6. Motor komutlarını güncelle
```

#### 4.2.2. Derinlik Kontrol Algoritması

Aracın su altında belirli bir derinlikte sabit kalmasını veya kontrollü bir şekilde derinlik değiştirmesini sağlayan algoritma:

```
1. Basınç sensörü ile mevcut derinliği ölç
2. Hedef derinlikten farkı hesapla
3. PID kontrol algoritması ile dikey motor gücünü hesapla
4. Derinlik kontrolünü sağlamak için motorları güncelle
```

### 4.3. Görev 2: Su Altı Kabloları Takibi ve Anomali Tespiti Algoritması

```
[Başlangıç] → [Kamera Kalibrasyonu] → [Kablo Tespiti] → [Kablo Takibi] → [Anomali Tespiti] → [Anomali Tanımlama ve Kaydetme] → [Tüm Kablo Hattı Tarandı mı?] → [Görev Tamamlandı]
```

![Görev 2 Algoritma Akış Diyagramı](https://placeholder/gorev2_akis)

#### 4.3.1. Kablo Takip Algoritması

Su altındaki kabloyu tespit edip takip etmek için geliştirilmiş görüntü işleme tabanlı algoritma:

```
1. Su altı kamerasını aktifleştir
2. Görüntüyü ön işleme (gürültü azaltma, kontrast ayarlama)
3. Kenar tespiti algoritmaları ile kablo sınırlarını belirle
4. Hough dönüşümü ile kablo yönünü tespit et
5. Aracı kablo yönü ile hizala
6. Kablonun takip edilmesi için motor komutlarını güncelle
```

#### 4.3.2. Anomali Tespit ve Tanımlama Algoritması

Kablo üzerindeki anomalileri (şekilleri) tespit etmek ve tanımlamak için geliştirilen yapay zeka destekli algoritma:

```
1. Eğitilmiş Konvolüsyonel Sinir Ağı (CNN) modelini yükle
2. Görüntü içinde nesne tespiti yap
3. Tespit edilen nesnenin özelliklerini çıkar
4. Önceden tanımlanmış şekillerle eşleştir
5. Tanımlanan anomaliyi kaydet (zaman, konum, şekil ismi, görüntü)
6. Araç ekranında tespit edilen anomali ismini göster
```

## 5. Algoritma Optimizasyonu ve Performans İyileştirmeleri

### 5.1. Hesaplama Kaynaklarının Verimli Kullanımı

Raspberry Pi 4'ün sınırlı kaynaklarını en verimli şekilde kullanmak için aşağıdaki optimizasyonlar yapılmıştır:

- Görüntü işleme için OpenCV kütüphanesinin optimizasyonları
- Paralel işlem kullanımı
- Gerçek zamanlı işlemler için önceliklendirme
- Yalnızca gerekli sensör verilerinin toplanması

### 5.2. Gürbüz Algoritma Tasarımı

Deniz altı ortamının zorluklarına karşı algoritmalarımızın dayanıklılığını artırmak için:

- Sensör verilerinde hata tespiti ve filtreleme
- Yedekli sistemlerle veri doğrulama
- Acil durum senaryoları için kurtarma algoritmaları
- Kötü görüş koşulları için adaptif görüntü işleme

## 6. Final Tasarım Seçimi ve Gerekçeleri

### 6.1. Navigasyon ve Kontrol Algoritmaları

Final tasarımda PID tabanlı kontrol algoritması seçilmiştir. Bu seçimin sebepleri:

- Uygulama kolaylığı ve hesaplama verimliliği
- Su altı ortamında test edilmiş ve ispatlanmış performans
- Raspberry Pi 4 üzerinde gerçek zamanlı çalışabilmesi
- Sistem kararlılığını sağlaması

### 6.2. Görüntü İşleme ve Nesne Tanıma

Anomali tespiti için CNN tabanlı bir yapay zeka modeli (ResNet-18) tercih edilmiştir. Bu seçimin sebepleri:

- Yüksek tanıma doğruluğu (test verilerinde %95 üzeri başarı)
- Raspberry Pi 4 üzerinde çalışabilecek optimizasyonlar
- Transfer öğrenme ile az veri setiyle eğitilebilmesi
- Su altı ortamındaki ışık değişimlerine karşı dayanıklılık

### 6.3. Sensör Füzyonu

Extended Kalman Filter (EKF) kullanılarak sensör verilerinin füzyonu sağlanmıştır. Bu seçimin sebepleri:

- Doğrusal olmayan sistemler için uygunluk
- Sensör gürültülerine karşı dayanıklılık
- Hesaplama verimliliği
- Konum tahminindeki yüksek doğruluk

## 7. Sonuç ve Gelecek Çalışmalar

Geliştirilen algoritma sistemi, Teknofest 2025 görevlerini başarıyla tamamlamak için gerekli tüm bileşenleri içermektedir. Gelecekte yapılacak iyileştirmeler:

- Daha gelişmiş makine öğrenmesi modelleri
- Batimetrik harita oluşturma yeteneği
- Çoklu araç koordinasyonu algoritmaları
- Daha verimli güç yönetimi algoritmaları

Bu algoritmalar, aracımızın otonom görevleri güvenli ve verimli bir şekilde tamamlamasını sağlayacak şekilde tasarlanmıştır.