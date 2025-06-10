# Yapay Sinir Ağları - Perceptron ile AND Kapısı

Bu proje, tek katmanlı perceptron kullanarak AND mantık kapısının nasıl öğrenileceğini göstermektedir. Perceptron, yapay sinir ağlarının en temel yapı taşlarından biridir ve doğrusal olarak ayrılabilir problemleri çözebilir.

## Kod Ne Yapar?

### 1. Perceptron Sınıfı
- **Ağırlık İlklendirme**: Rastgele ağırlıklar ve bias değeri ile başlar
- **Aktivasyon Fonksiyonu**: Step fonksiyonu kullanır (0 veya 1 çıkışı)
- **Öğrenme Algoritması**: Gradient descent ile ağırlıkları günceller

### 2. AND Kapısı Öğrenme Süreci
```
Girdi (X1, X2) → Çıktı (Y)
0, 0 → 0
0, 1 → 0
1, 0 → 0
1, 1 → 1
```

### 3. Eğitim Adımları
1. **Veri Hazırlama**: AND kapısı truth table'ı oluşturulur
2. **Forward Pass**: Girdi değerleri ağırlıklarla çarpılır ve toplanır
3. **Aktivasyon**: Step fonksiyonu ile çıktı belirlenir
4. **Hata Hesaplama**: Gerçek çıktı ile tahmin arasındaki fark bulunur
5. **Backward Pass**: Ağırlıklar ve bias güncellenir
6. **İterasyon**: Bu süreç hata sıfır olana kadar tekrarlanır

### 4. Matematiksel İşlemler
- **Net Girdi**: `net = w1*x1 + w2*x2 + bias`
- **Aktivasyon**: `output = 1 if net > 0 else 0`
- **Ağırlık Güncelleme**: `w_new = w_old + learning_rate * error * input`

## Beklenen Sonuç

Eğitim tamamlandığında perceptron:
- (0,0) girdisi için 0 çıkışı verir
- (0,1) girdisi için 0 çıkışı verir
- (1,0) girdisi için 0 çıkışı verir
- (1,1) girdisi için 1 çıkışı verir

## Öğrenme Çıktıları

Bu kod çalıştırıldığında:
- Perceptron'un temel çalışma prensibi anlaşılır
- Ağırlık güncelleme mekanizması görülür
- Doğrusal olarak ayrılabilir problemlerin çözümü öğrenilir
- Yapay sinir ağlarının temel yapı taşı kavranır

## Teknik Detaylar
- **Algoritma**: Perceptron Learning Rule
- **Aktivasyon Fonksiyonu**: Step Function
- **Öğrenme Oranı**: Ayarlanabilir (genellikle 0.1-0.01 arası)
- **Yakınsama**: Garantili (doğrusal ayrılabilir problemler için)

Bu basit örnek, karmaşık sinir ağı mimarilerinin temelini oluşturur ve makine öğrenmesinin çalışma prensiplerini anlamak için mükemmel bir başlangıç noktasıdır.
