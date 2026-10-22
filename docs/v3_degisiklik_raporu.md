# TR-LRS v3: eski veri setine göre neler değişti

Bu belge, Eylül 2026'daki ilk veri setinden (`tr-lrs-clean-ptm`, "v2")
Ekim 2026'daki `dataset/tr-lrs-v3`'e geçişte neyin bozuk olduğunu, nasıl
düzeltildiğini ve sonucun ne olduğunu anlatır. İngilizce teknik belgeler
`docs/dataset_construction.md` (v3'ün yapım yöntemi, makale için) ve
`docs/audit_and_roadmap_2026-10-17.md` (denetim ve günlük) dosyalarındadır;
buradaki her sayı oralardaki ölçümlerden alınmıştır.

## 1. Bir bakışta

| | v2 (2026-09-28) | v3 (2026-10-22) |
|---|---|---|
| Kaynak | 224x224 yüz klipleri (iki ayrı kırpma koşusunun tar dosyaları) | 39 tam kayıt, hepsi 1080p |
| Oturum | 33 (`video-N` ve meclis adları karışık) | 39 (19 meclis arşivi + 20 YouTube, tek adlandırma) |
| Transkript | Whisper `small`; `.txt`'ye yazılırken Whisper'ın güven sinyalleri atılmış | Whisper `large-v3` (faster-whisper); kelime güvenleri ve `no_speech_prob` kullanılıyor |
| İ / I | ikisi de `I` (kurtarılamaz sanılıyordu) | ayrı: 899 bin `İ`, 450 bin `I` |
| Klip | 218.091 → 154.191 tutuldu (%70,7) | 272.258 cümle → 225.977 tutuldu (%83,0) |
| Süre | 105,4 saat | 224,6 saat |
| Klip başına | ortanca 2,2 s, 4 kelime | ortanca 3,1 s, 6 kelime |
| Test seti | `2025-2026-birlesim-3` + `video-10`; birincisinin transkripti başka güne aitti | `2025-2026-birlesim-3` + `yt-NPJohShU_iw`, ikisi de doğru transkriptle |
| Konuşmacı | tek yüz; yakın planda dinleyici kesilebiliyordu | 4 yüze kadar takip, dudak hareketine göre seçim |
| PTM maskesi | kliplere yazılmış (her epoch aynı maske) | eğitimde uygulanıyor (her epoch yeni maske) |
| Yeniden üretim | tar dosyalarından, ilk aşama kod dışında | `scripts/corpus.py` ile baştan sona, kesintiden devam edebilir |

## 2. Eski veri setinde neler bozuktu

17 Ekim'de makinedeki her klasör okunarak yapılan denetimin bulguları:

1. **YouTube kayıtları 360p indirilmişti.** 20 YouTube oturumu (derlemin
   yarısı, v2 test oturumu `video-10` dahil) `yt-dlp -t mp4` ile 640x360
   inmişti; 1080p mevcuttu. 360p'de yüz 80–120 piksel genişliğindeydi, yani
   96x96 ağız kesiti aslında *büyütülmüş* bir görüntüydü.
2. **İ/ı ayrımı kırpıcıda kaybolmuştu.** Whisper JSON çıktısı doğru
   yazımdaydı (`İngilizce`, `Fırat`); kayıp, kırpıcıdaki tek bir
   `sentence.upper()` satırındaydı (`i` → `I`). Kelimelerin %54'ünde i veya ı
   var; v2 belgeleri bu kaybı "kurtarılamaz" diye kaydetmişti. Çözüm için
   sözlük gerekmiyordu, yalnızca JSON'dan yeniden kesmek.
3. **Whisper `small` bazı oturumlarda yoğun uyduruyordu.** `birlesim-94`'te
   18.238 segmentin 14.190'ı "Özgür Karabat" metniydi; 11.074 segmentin
   `no_speech_prob` değeri 0,5'in üzerindeydi. Bu sinyaller JSON'da vardı ama
   `.txt`'ye geçmemişti; v2'nin temizleme kuralları sorunu metin
   istatistiklerinden yeniden keşfetmek zorunda kalmıştı. v2'de kliplerin
   %27'si metin kurallarına, %10,6'sı yalnız tekrar/uydurma kurallarına
   gidiyordu.
4. **İki farklı kırpma koşusu vardı, ikisi de eksikti.** Koşu A (33 oturum,
   218.091 klip) v2'nin kaynağıydı; koşu B (39 oturum, daha katı takip) hiç
   içeri alınmamıştı ve 6 oturumu (~47 saat kaynak) v2'de hiç yoktu. İki
   koşu aynı klipte farklı kelime zamanları üretiyordu.
5. **Test oturumunun transkripti yanlış dosyadandı.** `video-35_ts.json`
   aslında `birlesim-5`'in ikinci bir transkripsiyonuydu; `birlesim-3`'ün
   gerçek transkripti `video-34_ts.json`'daydı. Her iki kırpma koşusu da
   `birlesim-3` için yanlış dosyayı kullanmıştı. v2'nin test oturumunda
   etiketler başka güne aitti; v2 test sayıları anlamsızdır.
6. **Kırpıcının diğer kusurları:** cümleler yalnız noktalama ve 10 s
   sınırına göre kesiliyordu (duraklama yoktu, uzun segmentler kelime
   ortasından bölünüyordu); video iki kez kodlanıyordu (OpenCV `mp4v` sonra
   libx264); kesintiden devam "sıradaki boş indeks" mantığıyla kopya klipler
   üretiyordu; her klip için ayrı `ffmpeg` süreci ve OpenCV seek yüzünden bir
   oturum 8–20 saat sürüyordu.
7. **`yt-fP99G1PONhQ`'nun eski transkripti çöptü** (her segment "HECK" veya
   "DEN"); sesi normaldi.
8. **Adlandırma karışıktı.** `video-1..41` adları hangi meclis oturumuna
   karşılık geldiğini söylemiyordu; eşleme dosya boyutundan çıkarıldı
   (denetim belgesi, bölüm 6).
9. **Disk.** Aynı videolar üç yerde (btrfs reflink) duruyordu; 465 GB'lık
   diskte 26 GB boştu ve yeniden kesim için önce yer açmak gerekti.

## 3. Nasıl düzeltildi

### Faz 0: disk ve kaynakların düzene sokulması (17 Ekim)

- 39 tam video kopyalar arasında bayt bayt doğrulanıp `videos/` altına tek
  adla alındı: meclis kayıtları `<dönem>-birlesim-<n>.mp4`, YouTube
  kayıtları `yt-<video id>.mp4`. `videos/sources.tsv` her kaydın kaynağını,
  çözünürlüğünü, süresini, boyutunu ve SHA-256'sını tutar.
- 20 YouTube oturumu 1080p H.264 olarak yeniden indirildi
  (`scripts/corpus.py download`; `yt-dlp` 2026.08+ ve `--js-runtimes node`
  gerekir, yoksa HTTP 403). Dosyalar 4–6 GB; indirme 15 GB boş alanın
  altında duraklar ve 360p dosyayı yalnız doğrulamadan sonra değiştirir.
- Kopyalar, eski deneyler (`exp/`), pip önbelleği, çöp kutusu ve `data/raw`
  silindi (kullanıcı onayıyla). Depo `v2-dataset-2026-09-28` etiketiyle
  donduruldu.

### Faz 1: yeniden başlatılabilir boru hattı (`avsr/corpus/`)

Tek giriş noktası `scripts/corpus.py`; her aşama oturum başına
`work/<oturum>/status.json`'a girdi ve seçeneklerin özetini (hash) yazar,
yalnız bunlar değişince yeniden çalışır.

| Aşama | Ne yapar | Devam etme birimi |
|---|---|---|
| `ingest` | ffprobe bilgileri ve sağlama → `sources.tsv` | |
| `transcribe` | faster-whisper `large-v3`, int8_float16, 5 ışın, kelime zamanları, Silero VAD, önceki pencereye koşullanma kapalı | 20 dakikalık parça (1.017 parça) |
| `segment` | transkript → `segments.tsv`: cümle sonu, 0,4 s'den uzun duraklama, 10 s sınırı en uzun duraklamadan bölünür, en az 2 kelime; metin kararları | saniyeler sürer |
| `cut` | tek sıralı çözme, 25 fps ızgarasına en yakın kareler, yüz bulma, konuşmacı seçimi, 96x96 ağız kesiti + 16 kHz mono ses tek mp4'te | cümle (`clips.tsv`'ye satır satır) |
| `build` | klip tablosu, bölünme, manifestler | |
| `verify` | 11 kontrol | |

Kesintiden devam ederken parça/cümle düzeyinde kaldığı yerden sürer;
seçenekler değişmişse yarım sonuç atılır (`clips.options` eki). Elektrik
kesintisi için işler iki kez durdurulup yeniden başlatıldı; hiçbir şey
tekrar işlenmedi.

### Transkripsiyon: `small` yerine `large-v3`

Aynı dakikanın iki transkripti karşılaştırıldı: `large-v3` okunabilir,
doğru bölünmüş cümleler verirken `small` bozuk metin veriyordu. RTX 2060'ta
int8_float16 ile gerçek zamanın ~10 katı hızda; 332 saat ses 17–20 Ekim
arasında bitti. Eski `small` JSON'ları `work/<oturum>/transcript.small.json`
olarak saklandı. Sonuç: tekrar/uydurma transkript oranı %10,6'dan %0,8'e,
düşük güven oranı %11,5'ten %2,3'e düştü.

### Metin kuralları

v2'nin R1–R6 kuralları (uydurma kalıp, iç döngü, patlama, patlamalı cümle,
sıkışmış kelime, düşük güven) aynen korundu; iki yenilik:

- **R7 `no_speech`**: Whisper'ın kendi "burada konuşma yok" olasılığı
  0,6'yı geçen cümleler atılır. v2'de bu sinyal yoktu.
- v2'nin "konuşma klip sonundan 1 s'den uzak bitiyor" kuralı kalktı; klipler
  artık kelimelerin etrafından kesildiği için gereksiz.

Normalizasyon artık Türkçe büyük harf kuralıyla çalışır (`i` → `İ`,
`ı` → `I`); alfabe 29 Türkçe harf + Q, W, X. `.txt` dosyaları Whisper'ın
özgün yazımını korur.

### Kesim: yüz bulma, konuşmacı seçimi, hız

- **MediaPipe 1080p karede 200–300 piksellik yüzleri kaçırıyordu.** İlk
  oturumun ilk 300 cümlesinin 167'si "yüz yok" diye reddedildi, oysa yüz net
  ve karşıdan görünüyordu. Sebep: FaceLandmarker'ın dedektörü girdiyi
  192x192'ye küçültür; 16:9 kare 192x108 olur, 250 piksellik yüz 25 piksele
  iner. Çözüm (`FaceFinder`): kare 540 satıra küçültülüp yarı örtüşen kare
  karolarla taranır (her 16 tespitte bir), bulunan her yüz tam çözünürlükte
  2,5 yüz boyu çevresinde yeniden tespit edilir. Tutulan oran 301 cümlede
  124'ten 276'ya çıktı.
- **Konuşmacı dudak hareketine göre seçilir.** Tek yüz ayarında MediaPipe
  yakın planda konuşmacının arkasındaki dinleyiciyi veriyordu. Artık 4 yüze
  kadar takip edilir; karelerin en az yarısında bulunan ve en uzun yüzün en
  az yarısı boyundaki adaylar arasından iç dudak açıklığının standart sapması
  en büyük olan seçilir. Tutulan kliplerin %87,8'inde tek aday, %12,2'sinde
  iki veya daha fazla aday vardı.
- **"Seçilen konuşmacı saklanan bölgenin dışında" hatası.** İlk tam kesimde
  cümlelerin %6,2'si (16.154) bu hatayla düştü: saklanan kare en uzun yüze
  ortalanıyordu, dudak kuralı başka (daha küçük) bir yüzü seçince ağız kare
  dışında kalıyordu; ayrıca sınır kontrolü gereğinden iki kat sıkıydı.
  Düzeltme: kontrol yarı genişlikte; seçilen yüz yine dışarıdaysa cümlenin
  kareleri seçilen yüze ortalanarak yeniden okunur (seek ile, ucuz). Yeni
  `cut --retry-errors` yalnız hatalı satırları yeniden keser. 16.154 hatanın
  15.934'ü kurtuldu; kalan 220'de (%0,1) ağız kadraj kenarına çok yakın.
- **Hız.** Profil: MediaPipe çıkarımı sürenin üçte ikisi, 1080p küçültme
  yedide biri. Yapılanlar: tespit her ikinci karede (aradakiler
  enterpolasyon; yörünge zaten Kalman ile düzleniyor), yalnız en uzun yüzün
  3 yüz boyu çevresi saklanır, tarama her 16 tespitte bir, MediaPipe Intel
  tümleşik GPU'da (OpenGL ES; NVIDIA kart bu iş için kullanılamıyor, 4
  süreç iGPU'yu doyuruyor), devam ederken dosya başından çözmek yerine seek.
  Sonuç: çalışan başına 9,5 → 18,4 cümle/dk; 7 çalışan (4 GPU + 3 CPU) ile
  transkripsiyon yanında toplam ~80 cümle/dk. Tüm kesim 17–21 Ekim arasında
  bitti.
- Kesinti nedeniyle `run`'ın atladığı iki oturum (`sources.tsv`'de hâlâ 360p
  görünüyordu) `ingest` yenilenip ayrıca kesildi. Ders: her indirmeden sonra
  `ingest`.

### Kırpma geometrisi (v2'den korunan)

Ağız merkezi 8 dış dudak noktasının ortalaması, Kalman + RTS geri geçişi ile
düzlenir; kare kenarı yüz boyunun 0,63 katı (Auto-AVSR'nin 96 piksel
kesitinin alanı; 567 klipte ölçülmüştü), cümle başına tek ölçek. Çıktı 96x96
renkli, libx264 crf 18, 25 fps; ses AAC 16 kHz mono aynı dosyada. Ayrı `.aac`
dosyası yok. Kesilen her cümlenin landmark'ları `work/<oturum>/landmarks/`
altında (2,2 GB); başka kırpma ayarlarıyla yeniden kesmek yüz tespiti
gerektirmez.

### Faz 3: derleme, doğrulama, tokenizer

- `build --holdout 2025-2026-birlesim-3 yt-NPJohShU_iw`: iki kaynaktan birer
  tam oturum test; diğer her oturumun son %5'i doğrulama (bitişik blok, çünkü
  komşu klipler aynı konuşmadan).
- `verify`: dosya bütünlüğü, bölünme tutarlılığı, manifestler, alfabe, hem
  `İ` hem `I` var mı, uydurma kalıp/döngü yok mu; 900 örnek klipte H.264 kare
  video + 16 kHz AAC akışları, kare sayısı, ses uzunluğu, etiket metni,
  kelime zamanları. Hepsi geçti.
- SentencePiece unigram 1000 ve 500 birimlik tokenizer'lar
  (`dataset/tr-lrs-v3-tokenizer/`).

### Faz 5: depo temizliği

- Silindi (git geçmişinde duruyor): `scripts/{extract_raw, build_dataset,
  audit_labels, verify_dataset, build_manifest}.py`, `tests/test_scripts.py`,
  `avsr/preprocess/{pipeline, ptm_bake}.py`, `manifest.py`'deki ham derlem
  yardımcıları.
- `configs/base.yaml` ve ince ayar config'leri `dataset/tr-lrs-v3`'ü
  gösterir; PTM eğitimde açık. A/B deney dosyaları (`roi`, `smoothing`,
  `ptm`) yeni `cut` komutlarına göre yazıldı.
- Depo dışına taşındı (`~/Desktop/archive/`): Auto-AVSR referans kodu
  (`AUTO_AVSR_REFERENCE` ortam değişkeni), makaleler, v2 logları ve örnek
  verisi, eski `lr_codes` klasörleri.
- v2 belgeleri `docs/archive/`'da; README, roadmap, findings, project_report,
  deployment güncellendi. 254 test geçiyor.

## 4. Saatlerin hesabı: 332 saat videodan 225 saat veri setine

| Adım | Saat | Açıklama |
|---|---|---|
| Kayıtların toplam süresi | 332,4 | 19 meclis arşivi 159,9 h + 20 YouTube 172,6 h |
| Whisper'ın bulduğu konuşma (≥2 kelimelik cümle aralıkları) | 271,0 | 272.258 cümle. Aradaki 61 saat: oturum araları, oylamalar, alkış, sessizlik, müzik, kelimeler arası duraklamalar ve tek kelimelik parçalar |
| Metin kurallarının elediği | −9,4 | düşük güven 5,2 h, konuşma yok 2,4 h, tekrar/uydurma 1,4 h, diğer 0,4 h |
| Video kontrollerinin elediği | −37,0 | yüz yok 22,7 h, yüz karelerin %50–80'inde 14,0 h, ağız kadraj dışında 0,2 h, küçük yüz ~0 |
| **Veri seti** | **224,6** | 225.977 klip, kayıtların %67,6'sı, bulunan konuşmanın %82,9'u |

Video kontrollerindeki kaybın büyüklüğü (%13) derlemin doğasından geliyor:
meclis yayını konuşma sırasında sık sık geniş plana, kürsüye veya
Başkanlık Divanı'na geçer; o sırada konuşan kişi görünmez. Bu cümleler
sesle birlikte vardır ama görsel bir karşılığı yoktur, dudak okuma için
kullanılamaz. Oturum başına tutulan oran %75 ile %89 arasında; v2'de bu
oran %15 ile %83 arasındaydı (iki oturum uydurma transkriptlerle doluydu).

v2 ile karşılaştırma: v2'de 218.091 klibin 154.191'i (105,4 h) tutulmuştu.
v3'te hem kaynak büyüdü (6 yeni oturum, +47 saat kayıt) hem kayıp küçüldü
(metin kayıpları %27'den %4'e), hem de cümleler daha uzun kesildi (ortanca
2,2 s → 3,1 s; 4 → 6 kelime), çünkü artık cümle sonu ve duraklamaya göre
kesiliyor, sabit uzunluk sınırına göre değil. Toplamda 2,1 kat saat.

## 5. Son veri seti

| | Eğitim | Doğrulama | Test | Toplam |
|---|---|---|---|---|
| Klip | 208.811 | 10.993 | 6.173 | 225.977 |
| Saat | 207,8 | 10,5 | 6,4 | 224,6 |
| Oturum | 37 | 37 | 2 | 39 |
| Kelime | 1.408.581 | 71.398 | 43.491 | 1.523.470 |
| Farklı kelime | 105.994 | 17.591 | 12.888 | 110.591 |
| Eğitimde görülmeyen kelime | | %4,3 | %4,5 | |

| Klip başına | En az | %5 | Ortanca | Ortalama | %95 | En çok |
|---|---|---|---|---|---|---|
| Kare | 12 | 25 | 78 | 89,5 | 222 | 252 |
| Saniye | 0,48 | 1,0 | 3,12 | 3,58 | 8,88 | 10,08 |
| Kelime | 2 | 2 | 6 | 6,74 | 16 | 35 |

- 214.831 farklı cümle; 110.591 farklı kelimenin 55.434'ü bir kez geçiyor.
- Tutulan kliplerde yüz, örneklenen karelerin %99,5'inde bulundu; kliplerin
  %94,4'ünde her karede. Yüz boyu kaynakta 167–346 piksel (5.–95. yüzdelik,
  ortanca 201), yani ağız bölgesi 105–218 kaynak pikselinden 96'ya
  küçültülüyor; 360p'deki gibi büyütme yok.
- %17,3 klipte baş belirgin dönük (yaw > 0,5).
- Kelime güveni ortancası 0,984; 5. yüzdelik 0,90.
- Diskte 14 GB, 451.954 dosya. Gzip arşivi: `tr-lrs-v3.tar.gz` (Drive'da,
  yanında `.sha256`).

Dosya düzeni (LRS2/LRS3 ile aynı mantık):

```
dataset/tr-lrs-v3/<oturum>/<no>.mp4   96x96 ağız, 25 fps (akış 0) + 16 kHz mono AAC (akış 1)
dataset/tr-lrs-v3/<oturum>/<no>.txt   Text:  <Whisper yazımıyla cümle>
                                      <kelime> <başlangıç> <bitiş> <güven>  ...
dataset/tr-lrs-v3-manifests/{train,val,test}.tsv
dataset/tr-lrs-v3-tokenizer/unigram{1000,500}.{model,vocab}
dataset/tr-lrs-v3.clips.tsv           272.258 cümlenin hepsi: karar, bölünme, kare, güven, tespit oranı, yaw, metin
dataset/tr-lrs-v3.settings.json       aşama seçenekleri ve bölünme
```

50 örnek klip (36 eğitim, 7 doğrulama, 7 test; 38 oturumdan) depoda
`samples/` altında, `samples/clips.tsv` ile birlikte.

## 6. Bilinen sınırlar ve sıradaki işler

- Transkriptler (test seti dahil) bir tanıyıcının çıktısı; ölçülen hata
  oranı Whisper'a göredir, söylenene göre değil. Birkaç yüz test klibinin
  elle düzeltilmesi mutlak doğruluk için tek yol.
- Konuşmacı seçimi birkaç düzine klipte gözle doğrulandı, ölçülmedi; iki
  veya daha fazla adaylı kliplerden (%12) bir örneklem incelenmeli.
- Aynı milletvekilleri birçok oturumda konuşuyor; test seti konuşmacı
  bakımından eğitimden bağımsız değil.
- Kayıtların %13'ünde konuşmacı görünmüyor; başka görüntü olmadan
  kurtarılamaz.
- v2'ye ait ince ayarlı kontrol noktaları eski alfabeyle eğitildi; v3 test
  setinde çıkış katmanları yeniden eğitilmeden değerlendirilemez.
- Sıradaki adım: v3 üzerinde eğitim (`configs/finetune/vsr_base.yaml`,
  uzak GPU; `docs/deployment.md`).
