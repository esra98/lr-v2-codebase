# Türkçe dudak okuma ve görüntü+ses modeli: kullanım

Bu rehber iki model paketi için de geçerlidir; hangisi olduğu
`model_info.json`'daki `modality` alanında yazar:

- **`tr-vsr-model`** (`video`): sessiz bir konuşma videosundan konuşanın
  dudak hareketlerini okuyup Türkçe metin çıkarır (görsel konuşma tanıma,
  VSR). Ses kullanılmaz.
- **`tr-avsr-model`** (`audiovisual`): dudak hareketleriyle birlikte
  videonun sesini de kullanır (görüntü+ses tanıma, AVSR). Gürültülü seste
  dudaklar yardımcı olur. Videoda ses akışı olmalıdır.

| | |
|---|---|
| Model | Auto-AVSR mimarisi: 3D konvolüsyon + ResNet-18 ön uç, 12 Conformer bloğu, 6 Transformer decoder bloğu, 241M parametre. Görüntü+ses: ek olarak 1D ResNet-18 ses ön ucu ve 12 Conformer bloğu, iki kolu birleştiren MLP; 435M parametre |
| Başlangıç | Ma ve ark. `vsr_trlrs2lrs3vox2avsp_base` (3.291 saat İngilizce); görüntü+ses modelinin ses kolu `asr_trlrs3vox2_base` (1.759 saat) |
| Fine-tune | TR-LRS v3: TBMM Genel Kurulu konuşmaları (TBMM TV / Meclis TV yayınları), 207,8 saat, 208.811 klip; görüntü+ses modelinde sese babble gürültüsü karıştırılarak |
| Çıktı | Büyük harfli Türkçe, noktalama yok, sayılar yazıyla (`1921` → `BİN DOKUZ YÜZ YİRMİ BİR`), İ/I ayrımı korunur |
| Doğruluk | `eval/test_<girdi>.json` (`test_video`, `test_audiovisual`, gürültülü testler `test_audiovisual_snr0` vb.): eğitimde hiç görülmemiş iki oturumda (6.173 klip) kelime (WER) ve karakter (CER) hata oranı |

## 1. Klasörde ne var

```
tr-vsr-model/  (ya da tr-avsr-model/)
  checkpoints/epoch-avg.ckpt   ağırlıklar (son 10 epoch'un ortalaması)
  config.yaml                  model ve eğitim ayarları
  tokenizer.model              1.000 birimlik Türkçe SentencePiece
  eval/test_*.json             test sonuçları (WER, CER), girdi moduna göre
  eval/test_*.tsv              her test klibi: referans ve modelin çıktısı
  csv/metrics*.csv             epoch başına val/cer, val/wer, loss (devam ettirilen eğitimde parça parça)
  logs/                        eğitim ve test logları
  model_info.json              ortalanan checkpoint'ler, modelin modu
  code.tar                     modeli çalıştıran kod (tr-lipreading)
  assets/face_landmarker.task  yüz noktaları modeli (MediaPipe)
  MODEL_KULLANIMI.md           bu dosya
```

## 2. Kurulum (bir kez)

Aşağıdaki komutlarda `tr-vsr-model` geçen yerlere, görüntü+ses modelinde
`tr-avsr-model` yazın.

Laptoptaki `~/Desktop/tr-lipreading` aynı kodu içeriyorsa (`git log`'da
`code.tar`'ın commit'i varsa) o kullanılabilir; 2b'ye geçin.

**2a. Kodu modelin yanından açmak:**

```bash
mkdir -p ~/tr-vsr && tar -xf tr-vsr-model/code.tar -C ~/tr-vsr
cp -r tr-vsr-model/assets ~/tr-vsr/
cd ~/tr-vsr
python3 -m venv .venv && source .venv/bin/activate
# PyTorch: GPU sürücüsüne uygun sürüm (ör. CUDA 12.6+ için cu126). GPU yoksa düz "pip install torch".
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt mediapipe opencv-python-headless
```

**2b. Mevcut proje:** `~/Desktop/tr-lipreading` içinde `.venv` zaten
MediaPipe ve OpenCV'yi içeriyor; `assets/face_landmarker.task` de orada.

## 3. Bir videoyu okumak

```bash
cd ~/tr-vsr            # ya da ~/Desktop/tr-lipreading
source .venv/bin/activate
python training/transcribe.py --run /tam/yol/tr-vsr-model konusma.mp4
```

Görüntü+ses modeli videonun sesini kendisi okur (`--run /tam/yol/tr-avsr-model`).

Çıktı (sekmeyle ayrılmış):

```
konusma.mp4	BUGÜN DE BU DURUŞUN EN GÜÇLÜ TEMSİLCİSİ ...
```

10 saniyeden uzun videolar 10 saniyelik parçalara bölünür, her parça
zamanıyla bir satır olur:

```
uzun.mp4	0.0-10.0	...
uzun.mp4	10.0-20.0	...
```

Seçenekler:

| Seçenek | Ne yapar |
|---|---|
| `video1.mp4 video2.mp4 ...` | Birden fazla video |
| `--chunk-seconds 8` | Parça uzunluğu (varsayılan 10). Eğitim klipleri en fazla ~10 saniyeydi (ortanca 3 saniye); daha uzun parçalar önerilmez |
| `--save-crop agiz.mp4` | Modelin gördüğü 96x96 ağız kırpıntısını kaydeder: yanlış kişi ya da kayık kırpma olup olmadığını görmek için |
| `--beam-size 10` | Daha hızlı, biraz daha az doğru (varsayılan 40) |
| `--cpu` | GPU olsa da CPU'da çalıştırır |
| `--mode video` | Görüntü+ses modelinde sesi kullanmaz (ses kolu sıfırla beslenir). Yalnızca `train.input_mode_probs` ile sessiz batch'ler de görmüş bir model için anlamlıdır; yoksa çıktı bozuk olur. Sessiz videolar için `tr-vsr-model` kullanın |

Video ne olmalı:

- Video boyunca tek bir kişi konuşmalı; önden ya da hafif yandan görünmeli.
  Konuşmacı, dataset yapılırken olduğu gibi seçilir: kadrajda kalan yüzler
  arasında dudakları en çok hareket eden (ve en büyüğün en az yarısı
  boyunda olan). Seçilen kişiyi `--save-crop` ile kontrol edin.
- Yüz yüksekliği en az ~50 piksel (alın-çene). Çözünürlük ve fps serbest:
  1080p geniş planlardaki küçük yüzler de döşemeli aramayla bulunur, video
  25 fps'e çevrilir, yüz eğitimdeki ölçeğe küçültülür.
- Hız (laptop, RTX 2060 + 16 çekirdek, ölçüldü): 7 saniyelik 1080p video
  ~8-10 saniye (yüz bulma CPU'da, model GPU'da) + bir kerelik ~5 saniye
  model yükleme. GPU'suz (`--cpu`) aynı video ~1 dakika.
- Model meclis konuşmalarıyla eğitildi: resmi, hitabet tarzı konuşmada en
  iyi sonucu verir. Günlük konuşma, farklı ışık ve kamera açılarında hata artar.
- Dudak okuma modeli (`tr-vsr-model`) ses kullanmaz. Görüntü+ses modeli
  videonun kendi ses akışını kullanır (16 kHz mono'ya çevrilir, görüntüyle
  aynı zaman çizgisinde); ayrı ses dosyası verilmez.

## 4. Python içinden

```python
import torch
from avsr.inference import load_trained, mouth_crops, transcribe

module, _, _ = load_trained("/tam/yol/tr-vsr-model")
module.to("cuda" if torch.cuda.is_available() else "cpu").eval()

crops, detection_rate = mouth_crops("konusma.mp4")   # uint8 [T, 96, 96, 3], 25 fps
for start, end, text in transcribe(module, crops, chunk_frames=250):
    print(f"{start:.1f}-{end:.1f}s  {text}")
```

Görüntü+ses modeli sesi de ister:

```python
from avsr.inference import sound_track

audio = sound_track("konusma.mp4")                   # float32 [S], 16 kHz
for start, end, text in transcribe(module, crops, chunk_frames=250, audio=audio):
    print(f"{start:.1f}-{end:.1f}s  {text}")
```

Ağız zaten kırpılmışsa (dataset klipleri gibi 96x96, 25 fps), yüz bulma
atlanır:

```python
from avsr.preprocess.video_io import read_video
crops = read_video("dataset/tr-lrs-v3/2025-2026-birlesim-3/00077.mp4")
print(transcribe(module, crops)[0][2])     # görüntü+ses: audio=sound_track(<aynı klip>)
```

## 5. Test setinde yeniden ölçmek

Dataset (`dataset/tr-lrs-v3`, manifest'ler, tokenizer) proje klasöründeyse:

```bash
python training/evaluate.py --run /tam/yol/tr-vsr-model
python training/evaluate.py --run /tam/yol/tr-avsr-model decode.snr_target=0   # 0 dB babble gürültüsüyle
```

Sonuç modelin `eval/test_<girdi>.json` ve `.tsv` dosyalarına yazılır.
`--split val` doğrulama setini ölçer.

## 6. Sonuçları okumak

- **WER** (kelime hata oranı): yanlış, eksik ve fazla kelimelerin referanstaki
  kelime sayısına oranı. **CER** aynısının karakter düzeyindeki karşılığı.
  Dudak okumada CER, WER'den belirgin düşüktür: kelimenin çoğu doğru, eki
  yanlış olabilir.
- `eval/test_video.tsv`, her klip için referansı ve modelin çıktısını yan
  yana gösterir; modelin nerede yanıldığını görmek için en iyi yer.
- Etiketler Whisper large-v3 ile otomatik üretildi; referanslarda da
  hatalar olabilir.

## 7. Sık karşılaşılanlar

| Belirti | Neden / çözüm |
|---|---|
| `MediaPipe model not found` | `assets/face_landmarker.task` proje klasöründe olmalı (`cp -r tr-vsr-model/assets .`) |
| `ModuleNotFoundError: cv2` / `mediapipe` | `pip install mediapipe opencv-python-headless` |
| `no face found` | Videoda yüz bulunamadı: çok küçük, çok yan ya da kapalı |
| `no audio stream` | Görüntü+ses modeline sessiz video verildi; `tr-vsr-model` kullanın |
| `a face was found in only 40% of the frames` | Yüz çoğu karede yok; çıktı güvenilmez |
| Çıktı hep aynı ya da boş | `--save-crop` ile kırpıntıya bakın: ağız ortada ve görünür mü |
| CPU'da yavaş | Beam search CPU'da ağırdır; `--beam-size 10` ya da GPU |
