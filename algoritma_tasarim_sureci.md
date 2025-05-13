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

## 3.Algoritma Entegrasyonu

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

<img width="1351" alt="Ekran Resmi 2025-05-13 09 43 49" src="https://github.com/user-attachments/assets/2856689b-a228-4b22-807d-9d68a4b90c2e" />


### 4.2. Görev 1: Kayıp Hazine Avı Algoritmasi

<img width="1543" alt="Ekran Resmi 2025-05-13 09 56 48" src="https://github.com/user-attachments/assets/0d8e1ff0-a0b8-4f67-bcc4-9966de445b76" />




#### 4.2.1. Derinlik Kontrol Algoritması

Aracın su altında belirli bir derinlikte sabit kalmasını veya kontrollü bir şekilde derinlik değiştirmesini sağlayan algoritma:

<img width="1177" alt="Ekran Resmi 2025-05-13 10 03 50" src="https://github.com/user-attachments/assets/31725798-01d9-4265-aefe-e47ae326738f" />


### 4.3. Görev 2: Su Altı Kabloları Takibi ve Anomali Tespiti Algoritması

<img width="1243" alt="Ekran Resmi 2025-05-13 10 09 52" src="https://github.com/user-attachments/assets/3c97e38c-9206-469b-aae9-77d4bee4d9b8" />


#### 4.3.1. Anomali Tespit ve Tanımlama Algoritması

Kablo üzerindeki anomalileri (şekilleri) tespit etmek ve tanımlamak için geliştirilen yapay zeka destekli algoritma:

<img width="1334" alt="Ekran Resmi 2025-05-13 10 15 48" src="https://github.com/user-attachments/assets/54f25a32-cd88-41a1-a252-3b2932f5e930" />

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

## 7. Sonuç ve Gelecek Çalışmalar

Geliştirilen algoritma sistemi, Teknofest 2025 görevlerini başarıyla tamamlamak için gerekli tüm bileşenleri içermektedir. Gelecekte yapılacak iyileştirmeler:

- Daha gelişmiş makine öğrenmesi modelleri
- Batimetrik harita oluşturma yeteneği
- Çoklu araç koordinasyonu algoritmaları
- Daha verimli güç yönetimi algoritmaları

Bu algoritmalar, aracımızın otonom görevleri güvenli ve verimli bir şekilde tamamlamasını sağlayacak şekilde tasarlanmıştır.
