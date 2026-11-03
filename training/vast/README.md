# vast.ai kurulumu: Türkçe dudak okuma ve ses+görüntü fine-tune'u

Bu paket (`lr-v2-vast.tar`), boş bir Ubuntu vast.ai sunucusunda tam
fine-tune'u başlatmak için gereken her şeyi içerir: kod, kurulum ve test
betikleri, Drive linkleri ve sha256 değerleri. Büyük dosyalar (dataset,
Python kütüphaneleri, pretrained ağırlıklar) paketin içinde değildir;
`setup.sh` onları Google Drive'dan **gdown** ile indirir ve doğrular. Drive
klasörü linkle paylaşılmış olduğu için sunucuda Google hesabı ya da anahtar
gerekmez.

İki model eğitilebilir; betikler `MODE` ortam değişkeniyle seçer:

| `MODE` | Model | Girdi | Config | Deney adı | Model paketi |
|---|---|---|---|---|---|
| `video` (varsayılan) | dudak okuma (VSR), 241M parametre | ağız kırpıntısı | `training/configs/finetune/vsr_base.yaml` | `vsr_finetune` | `tr-vsr-model.tar` |
| `audiovisual` | görüntü+ses (AVSR), 435M parametre | ağız kırpıntısı ve klibin sesi | `training/configs/finetune/avsr_base.yaml` | `avsr_finetune` | `tr-avsr-model.tar` |

İkisi aynı sunucuda arka arkaya eğitilebilir (tek GPU olduğu için aynı anda
değil). Görüntü+ses modeli, görüntü modelinin tarifini ve başlangıç
ağırlıklarını kullanır; üstüne Auto-AVSR'ın ses modelinden başlayan bir ses
kolu ve iki kolu birleştiren bir MLP eklenir. Böylece iki sonuç doğrudan
karşılaştırılabilir.

Akış:

```
laptop ──scp──> sunucu: lr-v2-vast.tar (~60 MB)
sunucu: setup.sh   (Drive'dan indir, doğrula, aç, kur)        ~15-30 dk
sunucu: [MODE=audiovisual] check.sh   (pytest, preflight, GPU bellek ölçümü)   ~15 dk
sunucu: [MODE=audiovisual] train.sh full                       günler
sunucu: finish.sh  (test seti, model paketi)                   ~30-60 dk
sunucu ──scp──> laptop: tr-vsr-model.tar / tr-avsr-model.tar   → sunucuyu kapat (Destroy)
```

## 0. Sunucuyu kiralarken

| Ayar | Değer | Neden |
|---|---|---|
| Image | `ubuntu:22.04` (veya 24.04), launch mode **SSH** | Kurulum kendi Python'unu getirir; başka bir şey gerekmez |
| GPU | Tek GPU, **en az 48 GB** (A6000, L40S, A100 80GB, H100); görüntü+ses için **80 GB önerilir** | 1.800 frame'lik batch 241M parametreli modelle; 24 GB kartta `FRAMES=900 ACCUM=2` gerekir. Görüntü+ses modeli (435M, iki Conformer) yaklaşık iki kat bellek ister; 48 GB'ta büyük olasılıkla `FRAMES=900 ACCUM=2` (ölçülmedi, `check.sh` ölçer) |
| Max CUDA | **≥ 12.8** | Kütüphaneler PyTorch 2.13 + CUDA 12.9 ile derlendi |
| Disk | **≥ 100 GB** | Dataset 13+14 GB, kütüphaneler 11 GB, checkpoint'ler ~15 GB (görüntü+ses ~25 GB; ikisi birden ~40 GB) |
| RAM | ≥ 32 GB | Veri yükleyiciler ~0,5 GB/işçi; eğitim süreci 3-4 GB |
| CPU | ≥ 16 çekirdek | Videolar CPU'da açılır; az çekirdek GPU'yu bekletir |
| İnternet indirme | ≥ 500 Mbps | Drive'dan 18 GB indirilecek |

Birden fazla GPU'lu sunucu almayın: kod tek GPU kullanır.

## 1. Sunucuya ne indirilecek

`setup.sh` aşağıdaki dosyaları sırayla, Drive ID'leriyle indirir
(`DRIVE_FILES.txt`). 1-7 aynı Drive klasöründedir:
<https://drive.google.com/drive/folders/16tybAQsivB4GPGB2piDDpuoSJeWToBCS>.
8, Auto-AVSR'ın kendi herkese açık linkidir (model zoo,
<https://github.com/mpc001/auto_avsr>); yalnızca görüntü+ses modunda gerekir.

| # | Dosya | Boyut | Ne | Sunucuda nereye |
|---|---|---|---|---|
| 1 | [`tr-lrs-v3.tar.gz`](https://drive.google.com/file/d/15dwNvrNDL_YJbRYSsC7T_Vnk2Ng25jwC/view) | 12,9 GB | TR-LRS v3: 225.977 ağız kırpıntısı klip (96x96, 25 fps), manifest'ler, tokenizer | açılır: `tr-lipreading/dataset/`; tar silinir |
| 2 | [`vsr_trlrs2lrs3vox2avsp_base.pth`](https://drive.google.com/file/d/162axOfnnZxVfhpbm9XK4SKAo_l3Qdpf_/view) | 1,0 GB | Auto-AVSR (Ma ve ark.) görsel konuşma tanıma ağırlıkları, başlangıç noktası | `tr-lipreading/pretrained/` |
| 3 | [`wheelhouse-py312-cu129.tar.part-00`](https://drive.google.com/file/d/1rAwn_Q4PhS60OLC3X9rJExzO6ZulEf98/view) | 900 MB | Wheelhouse, parça 1/5 (aşağıya bakın) | birleştirilir, kurulur, silinir |
| 4 | [`wheelhouse-py312-cu129.tar.part-01`](https://drive.google.com/file/d/1kx0uzLgmr1r-ENWgRrgJSMuXVbHhq9D0/view) | 900 MB | Wheelhouse, parça 2/5 (aşağıya bakın) | birleştirilir, kurulur, silinir |
| 5 | [`wheelhouse-py312-cu129.tar.part-02`](https://drive.google.com/file/d/1VFzwmBCiNUcHOTag0h4LMo1due8beG-D/view) | 900 MB | Wheelhouse, parça 3/5 (aşağıya bakın) | birleştirilir, kurulur, silinir |
| 6 | [`wheelhouse-py312-cu129.tar.part-03`](https://drive.google.com/file/d/1b--7Hs5Oi8cPh1hpj2F5ehzjjjx9Ymqb/view) | 900 MB | Wheelhouse, parça 4/5 (aşağıya bakın) | birleştirilir, kurulur, silinir |
| 7 | [`wheelhouse-py312-cu129.tar.part-04`](https://drive.google.com/file/d/1_u2b3SYTsp5PjX9oRBi0PidNM2TCpZsG/view) | 468 MB | Wheelhouse, parça 5/5 (aşağıya bakın) | birleştirilir, kurulur, silinir |
| 8 | [`asr_trlrs3vox2_base.pth`](https://drive.google.com/file/d/1YN9lwZN6iWn2qNQRpfpGpnf2r6ZTQqVT/view) | 928 MB | Auto-AVSR ses tanıma (ASR) ağırlıkları (LRS3+VoxCeleb2, 1.759 saat, LRS3'te %1,0 WER): görüntü+ses modelinin ses kolu | `tr-lipreading/pretrained/` |

Wheelhouse 4,3 GB'lık tek bir tar (`wheelhouse-py312-cu129.tar`): 63
Python paketi (wheel), PyTorch 2.13.0+cu129 dahil. Drive'a yüklenirken
kesilmemesi için 5 parçaya bölündü (`split -b 900M`); `setup.sh`
parçaları `cat` ile birleştirir ve tam tar'ı doğrular.

sha256 (`SHA256SUMS.drive`):

```
143122937a84a16748a8b22b415d3d49b993c522f44a2b8842eb46d1dda55b8e  tr-lrs-v3.tar.gz
bc8f44ecf8d5218c23aa5ed3675350457758927b572d74675c7afa1325c06e03  wheelhouse-py312-cu129.tar
fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb  vsr_trlrs2lrs3vox2avsp_base.pth
9b2abf4904f417d9c49fb8bc4a5b85fce3ba47b58c312634c417d7da2b102961  asr_trlrs3vox2_base.pth
```

`vsr_trlrs2lrs3vox2avsp_base.pth`, Auto-AVSR'ın yayımladığı dosyanın
birebir kopyasıdır (sha256 aynı). Auto-AVSR hazır bir görüntü+ses ağırlığı
yayımlamadığı için görüntü+ses modeli bu iki dosyadan başlar.

Paketin kendi içinden gelenler (indirme yok):

| Dosya | Ne |
|---|---|
| `code.tar` | Kod, git commit'i `COMMIT` dosyasında. `tr-lipreading/` olarak açılır |
| `assets/` | `babble_noise.wav` (ses gürültüsü), `face_landmarker.task` (yüz noktaları; yalnızca model kullanımında gerekir) |
| `bin/python-3.12.tar.gz` | Taşınabilir Python 3.12.16 (python-build-standalone); sistem Python'u kullanılmaz |
| `bin/gdown-wheels/` | gdown 6.5.1 ve bağımlılıkları (12 wheel), internetsiz kurulur |
| `DRIVE_FILES.txt` | İndirilecek dosyaların Drive ID'leri |
| `requirements-lock.txt` | Kurulan paketlerin kesin sürümleri |
| `SHA256SUMS.drive` | Drive dosyalarının sha256 değerleri |
| `SHA256SUMS.bundle` | Paketin içindeki dosyaların sha256 değerleri |
| `REMOTE_RUN.md` | Sunucuda çalıştırmanın kısa yolu: kirala, gönder, kur, başlat, izle, indir |
| `backup.sh`, `restore.sh` | Laptopta çalışır: sunucudaki eğitimin yedeğini çeker; yeni sunucuya gönderip sürdürür (8. bölüm) |
| `MODEL_KULLANIMI.md` | Eğitilen modeli kullanma rehberi; `finish.sh` model paketine koyar |

Sunucuda son durum:

```
$WORK/                         /workspace (yoksa /root)
  tr-lipreading/              proje
    avsr/ dataset-creation/ training/ tests/ docs/ deploy/
    assets/
    dataset/tr-lrs-v3/         39 oturum klasörü, 451.954 dosya
    dataset/tr-lrs-v3-manifests/  train.tsv val.tsv test.tsv
    dataset/tr-lrs-v3-tokenizer/  unigram1000.model
    pretrained/vsr_trlrs2lrs3vox2avsp_base.pth
    pretrained/asr_trlrs3vox2_base.pth   (görüntü+ses modu)
    .venv/                     Python 3.12 + kütüphaneler
    exp/vsr_finetune/          eğitim çıktısı (checkpoint'ler, metrikler)
    exp/avsr_finetune/         görüntü+ses eğitiminin çıktısı
    logs/
  python/                      taşınabilir Python
  tools/                       gdown
  downloads/                   wheelhouse tar'ı, pretrained'in kopyası
```

## 2. Laptoptan sunucuya yükleme

vast.ai panelindeki "Connect" düğmesi SSH komutunu verir:
`ssh -p <port> root@<adres>`. Aynı port ve adresle:

```bash
cd ~/Desktop
sha256sum -c lr-v2-vast.tar.sha256
scp -P <port> lr-v2-vast.tar root@<adres>:/root/
```

## 3. Kurulum

```bash
ssh -p <port> root@<adres>
cd /root && tar -xf lr-v2-vast.tar && cd lr-v2-vast
bash setup.sh
```

`setup.sh` altı adımı sırayla yapar ve her adımı bitince işaretler. Bir
yerde durursa sorunu giderip yeniden çalıştırın; bitmiş adımlar atlanır,
yarım kalan indirme kaldığı yerden sürer.

1. **Kontrol:** `nvidia-smi`, sürücünün CUDA sürümü, boş disk
2. **Python ve gdown:** taşınabilir Python 3.12, gdown paketin içinden
3. **İndirme:** 1. bölümdeki dosyalar gdown ile; wheelhouse parçaları
   birleştirilir; `sha256sum -c SHA256SUMS.drive`
4. **Açma:** kod, assets, ağırlıklar; dataset açılır, oturum (39) ve dosya
   (451.954) sayısı kontrol edilir, dataset tar'ı silinir
5. **Ses ağırlıkları:** `asr_trlrs3vox2_base.pth` (yalnızca görüntü+ses
   modunda gerekir). İnmezse kurulum durmaz, uyarı verir; görüntü modu
   etkilenmez. Sonra `bash setup.sh` yalnızca bu adımı tekrarlar
6. **Kütüphaneler:** `.venv` → wheelhouse'tan internetsiz `pip install`;
   sonunda PyTorch'un GPU'da işlem yaptığı denenir

Son satır `KURULUM TAMAM` olmalı. `KURULUM TAMAM (ses ağırlıkları eksik:
yalnızca görüntü modu)` görüntü modeli için yeterlidir.

Elle yapmak isterseniz eşdeğer komutlar:

```bash
mkdir -p /workspace/downloads && cd /workspace
tar -xzf ~/lr-v2-vast/bin/python-3.12.tar.gz                    # /workspace/python
python/bin/python3 -m venv tools
tools/bin/pip install --no-index --find-links ~/lr-v2-vast/bin/gdown-wheels gdown
cd downloads
grep -v '^#' ~/lr-v2-vast/DRIVE_FILES.txt | while read -r id name; do
  /workspace/tools/bin/gdown "$id" -O "$name" --continue --retries 5
done
cat wheelhouse-py312-cu129.tar.part-* > wheelhouse-py312-cu129.tar
sha256sum -c ~/lr-v2-vast/SHA256SUMS.drive                      # üçü de OK
mkdir -p /workspace/tr-lipreading/pretrained && cd /workspace/tr-lipreading
tar -xf ~/lr-v2-vast/code.tar && cp -r ~/lr-v2-vast/assets .
cp /workspace/downloads/vsr_trlrs2lrs3vox2avsp_base.pth pretrained/
cp /workspace/downloads/asr_trlrs3vox2_base.pth pretrained/     # görüntü+ses modu
tar -xzf /workspace/downloads/tr-lrs-v3.tar.gz                  # dataset/
/workspace/python/bin/python3 -m venv .venv
mkdir -p /workspace/downloads/wh && tar -xf /workspace/downloads/wheelhouse-py312-cu129.tar -C /workspace/downloads/wh
.venv/bin/pip install --no-index --find-links /workspace/downloads/wh -r ~/lr-v2-vast/requirements-lock.txt
```

## 4. Testler

```bash
bash check.sh                      # görüntü modeli
MODE=audiovisual bash check.sh     # görüntü+ses modeli
```

İki modeli de eğitecekseniz ikisini de çalıştırın; preflight ve bellek
ölçümü modele göre değişir. Çıktı `logs/check.log` /
`logs/check_audiovisual.log` dosyasına da yazılır.

| Test | Geçti sayılır | Süre |
|---|---|---|
| `pytest` | **0 failed**; ~205 passed. Bazıları `skipped`: referans kod, OpenCV ve ffmpeg yalnızca dataset yapılan makinede var | 5-10 dk |
| `training/preflight.py` | Son satır `READY TO TRAIN`. Her klip diskte mi, 500x3 klip açılıyor mu (görüntü+ses modunda ses de), 96x96 mı, tokenizer, pretrained ağırlıklar (görüntü 762 tensör; görüntü+ses 762 + 604), disk, 20 adım gerçek eğitim | 3-5 dk |
| `training/benchmark_memory.py` | 1.800 frame satırında `peak GB`, GPU belleğinin ~%80'inden az | 2 dk |

Bellek sonucuna göre (görüntü+ses için başına `MODE=audiovisual` ekleyin;
`check.sh` sonunda komutu kendisi yazar):

| 1.800 frame sığıyorsa | `bash train.sh full` |
|---|---|
| Yalnız 900 sığıyorsa | `FRAMES=900 ACCUM=2 bash train.sh full` |
| Yalnız 600 sığıyorsa | `FRAMES=600 ACCUM=3 bash train.sh full` |

## 5. Eğitim

```bash
bash train.sh full                     # görüntü modeli; arka planda başlar, SSH kapansa da sürer
MODE=audiovisual bash train.sh full    # görüntü+ses modeli
bash train.sh status        # en son başlatılan eğitim: son checkpoint zamanı, epoch başına val/cer, GPU, hata satırları
tail -f /workspace/tr-lipreading/logs/vsr_finetune.log     # görüntü+ses: logs/avsr_finetune.log
```

`status`, `resume` ve `finish.sh`, `MODE` ya da `EXP` verilmezse en son
`full` ile başlatılan eğitime bakar (`logs/last_exp`). Bir eğitim
çalışırken ikincisi başlatılamaz; ikinci model, birincisi bitince (isterseniz
önce `finish.sh` ile paketlenip indirildikten sonra) başlatılır.

Tarif (`training/configs/finetune/vsr_base.yaml`): 75 epoch, yeni katmanlar lr 1e-3,
pretrained kısımlar lr 2e-5 (ilk epoch sabit tutulur), AdamW, 5 epoch
warm-up + cosine, 1.800 frame/batch, 0,1 CTC + 0,9 attention, eğitimde
Partition-Time Masking. 241M parametre. Ayrıntı: `docs/deployment.md` 8. bölüm.

Görüntü+ses tarifi (`training/configs/finetune/avsr_base.yaml`): yukarıdakinin aynısı,
üstüne ses kolu (1D ResNet-18 + 12 Conformer bloğu, `asr_trlrs3vox2_base.pth`
ağırlıklarıyla) ve iki kolun çıktısını birleştiren MLP (8.192 gizli birim,
rastgele başlar, yeni katmanlarla aynı lr 1e-3). Decoder görüntü
ağırlıklarından gelir. Eğitimde sese babble gürültüsü karıştırılır (-5..20 dB
ya da temiz, Auto-AVSR gibi). 435M parametre. Bir epoch görüntü modelinden
uzun sürer (ikinci Conformer ve sesin açılması; ölçülmedi).

Ne beklenmeli (laptopta 8.000 klipte ölçüldü, 1 Kasım 2026):

| Ne zaman | Beklenen |
|---|---|
| Log başı | `Loaded 762 tensors` (görüntü+ses: ardından `Loaded 604 tensors`), sonra iki öğrenme oranı satırı (1e-3 ve 2e-5) |
| 1. epoch sonu | `val/cer` 0,9'un altında (laptop denemesinde 3.675 güncellemeden sonra 0,68). Görüntü+ses modelinde ses sayesinde belirgin daha düşük olmalı (ölçülmedi) |
| Sonraki epoch'lar | `val/cer` düşer |
| `val/cer` 5 epoch boyunca 0,9 üstünde | Eğitim öğrenmiyor: durdur (`pkill -f training/train.py`) |
| `status`'ta `amp_scale` sürekli düşüyor | 16-bit taşma: durdur, `bash train.sh resume train.precision=bf16-mixed` değil, baştan `EXP=vsr_bf16 bash train.sh full train.precision=bf16-mixed` |
| Logda `out of memory` / OOM | `FRAMES`'i küçült, `ACCUM`'u büyüt (çarpımları 1.800) |

Bir epoch'un süresini ilk epoch bitince `status` gösterir (adım sayısı ve
zaman). Toplam süre ≈ 75 x epoch süresi.

Bağlantı veya sunucu kesilirse:

Spot sunucu ve laptopa yedek: 8. bölüm.

```bash
bash train.sh resume        # en son başlatılan eğitim, last.ckpt'den, aynı ayarlarla sürer (epoch ortasından da)
EXP=avsr_finetune bash train.sh resume    # adıyla
```

İsteğe bağlı pilot (~2 saat): pretrained lr ölçeği 0,02 ve 0,1'in kısa
karşılaştırması. 0,02 küçük batch'lerle seçildi, 1.800 frame'de
doğrulanmadı.

```bash
bash train.sh pilot
MODE=audiovisual bash train.sh pilot
```

İsteğe bağlı, tek modelle iki mod: `MODE=audiovisual bash train.sh full
"train.input_mode_probs=[0.25,0,0.75]"` batch'lerin dörtte birini sessiz
(yalnız görüntü) eğitir; aynı model hem sesli hem sessiz videoda
kullanılabilir. Doğrulama üç modu ayrı raporlar ve en iyi checkpoint'ler
`val/cer_video`'ya göre seçilir. Denenmedi; ayrı iki model daha güvenli yol.

## 6. Bitirme ve indirme

Eğitim bitince (`train.sh status` → `eğitim çalışmıyor`, 75 epoch):

```bash
bash finish.sh                      # en son başlatılan eğitim
MODE=audiovisual bash finish.sh     # ya da modu/adı açıkça: EXP=vsr_finetune bash finish.sh
SNR="0 -5" bash finish.sh           # görüntü+ses: ek olarak 0 ve -5 dB babble gürültüsüyle test
```

1. Test seti (eğitimde hiç görülmemiş 2 oturum, 6.173 klip) beam search ile,
   son 10 checkpoint'in ortalamasıyla değerlendirilir:
   `exp/<deney>/eval/test_<girdi>.json` (WER, CER) ve `.tsv` (her klibin
   referansı ve modelin çıktısı). Görüntü modeli: `test_video`. Görüntü+ses
   modeli: `test_audiovisual`, `SNR` verildiyse `test_audiovisual_snr0` vb.
   (her gürültü düzeyi bir test süresi daha). `input_mode_probs` ile sessiz
   batch'ler de görmüşse `test_video` da yazılır
2. `tr-vsr-model/` ya da `tr-avsr-model/` klasörü yapılır ve kendi kendine
   yükleyip bir test klibini okuyarak denenir (görüntü+ses modelinde klibin
   sesiyle)
3. `/workspace/tr-vsr-model.tar` (~1 GB) ya da `/workspace/tr-avsr-model.tar`
   (~1,8 GB) ve `.sha256` yazılır

İki model de eğitildiyse `finish.sh`'yi her biri için çalıştırın. Laptopta
(görüntü+ses için adlarda `tr-avsr-model`):

```bash
scp -P <port> 'root@<adres>:/workspace/tr-vsr-model.tar*' ~/Desktop/
cd ~/Desktop && sha256sum -c tr-vsr-model.tar.sha256 && tar -xf tr-vsr-model.tar
```

`sha256sum` `OK` dedikten sonra vast.ai panelinden sunucuyu **Destroy**
edin. Stop, diski saklamaya devam eder ve ücretlendirir.

Modeli kullanma: `tr-vsr-model/MODEL_KULLANIMI.md` (görüntü+ses:
`tr-avsr-model/MODEL_KULLANIMI.md`).

`tr-vsr-model.tar` / `tr-avsr-model.tar` içinde:

| Dosya | Ne |
|---|---|
| `checkpoints/epoch-avg.ckpt` | Son 10 epoch'un ortalama ağırlıkları |
| `config.yaml` | Eğitim ayarları (model mimarisi buradan kurulur) |
| `tokenizer.model` | 1.000 birimlik Türkçe SentencePiece |
| `eval/test_*.json`, `.tsv` | Test sonuçları (girdi moduna ve gürültüye göre birer dosya) |
| `csv/metrics*.csv`, `logs/` | Eğitim boyunca metrikler, eğitim ve test logları |
| `model_info.json` | Hangi checkpoint'ler ortalandı |
| `code.tar`, `assets/` | Modeli çalıştıran kod ve yüz bulucu modeli |
| `MODEL_KULLANIMI.md` | Kullanım rehberi |

## 7. Sorun giderme

| Belirti | Çözüm |
|---|---|
| gdown `Too many users have viewed or downloaded this file recently` | Drive'ın dosya başına indirme kotası; birkaç saat sonra `bash setup.sh` (kaldığı yerden sürer) |
| gdown `Cannot retrieve the public link` / `Access denied` | Drive klasörünün paylaşımı "Bağlantıya sahip olan herkes" olmalı |
| `UYARI: asr_trlrs3vox2_base.pth indirilemedi` | Auto-AVSR'ın linki kotaya takıldı. Birkaç saat sonra `bash setup.sh` (yalnızca bu adım), ya da dosyayı laptopta indirip `scp -P <port> asr_trlrs3vox2_base.pth root@<adres>:/workspace/tr-lipreading/pretrained/` ve sunucuda `touch /workspace/.setup-done/audio-weights`. Kalıcı çözüm: dosyayı kendi Drive klasörüne kopyalayıp yeni ID'yi `DRIVE_FILES.txt`'ye yazmak |
| preflight `FAIL  pretrained audio weights exist` | Yukarıdaki satır: ses ağırlıkları eksik |
| sha256 `FAILED` | `rm -f /workspace/.setup-done/download /workspace/downloads/<dosya>`, `bash setup.sh` |
| `PyTorch GPU'yu göremiyor` | Sürücü CUDA 12.8'den eski: başka sunucu seç |
| Disk doldu | `df -h /workspace`; `downloads/wheelhouse-py312-cu129.tar` silinebilir |

## 8. Spot (kesilebilir) sunucu: kesinti, yedek, başka sunucuda sürdürme

Spot sunucu, daha yüksek teklif gelince durdurulur. Üç katman var:

| Katman | Nerede | Ne yapar | Kesintide kaybolan |
|---|---|---|---|
| `last.ckpt` her 20 dakikada | sunucu | eğitim, epoch sonunu beklemeden `last.ckpt` yazar (`train.checkpoint_every_minutes`) | en fazla 20 dk |
| `train.sh auto` | sunucu | aynı makine geri gelince eğitimi kendisi sürdürür (vast "onstart" komutu) | — |
| `backup.sh` | laptop | her `INTERVAL` dakikada sunucunun `exp/` ve `logs/` klasörlerini laptopa çeker | makine hiç geri gelmezse en fazla `INTERVAL` + 20 dk |

Epoch'un ortasından sürdürülen eğitim, kesintisiz eğitimle aynı batch'leri
aynı sırayla görür ve aynı adımda biter (`tests/test_training.py`,
`test_interrupted_run_resumes_in_the_middle_of_the_epoch`). Logdaki
`dataloader is not resumable` uyarısı bu yüzden zararsızdır.

**Sunucuyu kiralarken:** "On-start script" alanına

```bash
[ -f /root/lr-v2-vast/.work ] && bash /root/lr-v2-vast/train.sh auto >> /root/onstart.log 2>&1
```

`auto`, yarıda kalmış bir eğitim varsa (son `last.ckpt`, bitince yazılan
`metrics.json`'dan yeniyse) sürdürür, yoksa hiçbir şey yapmaz.

**Laptopta yedek** (sunucuda `setup.sh` rsync'i kurar; laptopta `rsync`,
`python3` ve vast'a kayıtlı SSH anahtarı gerekir):

```bash
bash backup.sh <instance id>               # adres ve durum her turda vast API'sinden
INTERVAL=60 bash backup.sh <instance id>   # dakika; varsayılan 120
bash backup.sh ssh5.vast.ai 12345          # ya da adres ve port
```

Terminal kapanınca durmaması için `tmux` / `screen` içinde ya da
`nohup bash backup.sh <id> &` ile çalıştırın. Yedekler `~/lr-v2-yedek/`
altında (`BACKUP_DIR`): `snapshots/<zaman>/` son 3 kopya (`KEEP`; değişmeyen
dosyalar hardlink, yer kaplamaz), `current` en son sağlam kopya, `backup.log`.
Bir kopya ancak her `.ckpt` sağlamsa ve sunucudaki `last.ckpt` yedektekinden
eski değilse `current` olur. Sunucuya ulaşılamazsa masaüstü bildirimi verir
ve vast'taki durumu yazar.

**Ne sıklıkla:** her tur değişen `last.ckpt`'yi indirir (görüntü modeli
~3 GB, görüntü+ses ~5 GB, ağırlıklar ve optimizer durumu), ayrıca her epoch'un
yeni checkpoint'ini (~1 / ~1,7 GB). Sunucunun dışarı trafiği ücretlidir
(RTX PRO 5000 teklifinde 20 $/TB): 120 dakikada bir ~0,7 $/gün, 30 dakikada
bir ~3 $/gün (görüntü modeli). Aynı makine geri gelirse laptop yedeği hiç
gerekmez; yalnızca makine kaybolursa kullanılır. Bu yüzden 60-120 dakika
yeterlidir.

**Makine geri gelmezse:** yeni (tercihen aynı GPU türünde) bir sunucu kirala,
paketi gönder, `bash setup.sh` (KURULUM TAMAM), sonra laptopta:

```bash
bash restore.sh <yeni instance id>     # current'ı gönderir ve train.sh resume
```

`restore.sh` gönderirken eğitim çalışıyorsa ya da sunucuda aynı deneyin
checkpoint'i varsa durur (`FORCE=1` ile üzerine yazar). Gönderilen ~15 GB
(görüntü+ses ~25 GB): laptopun yükleme hızına göre sürer. Farklı GPU türünde
`FRAMES` ve `ACCUM` eski haliyle kalır (`logs/<deney>.opts`); sığmazsa
`train.sh resume` sonuna `data.max_frames_per_batch=900
train.accumulate_grad_batches=2` gibi ayarlar eklenebilir, çarpımları aynı
kalmalı. Sonra laptopta yedeği yeni sunucuya çevir:
`bash backup.sh <yeni instance id>`.

## Paketin yeniden yapılması (laptopta)

Kod değişirse, commit edip:

```bash
bash training/vast/make_bundle.sh ~/Desktop/vast-bundle-build ~/Desktop/lr-v2-vast.tar
```

`~/Desktop/vast-bundle-build/bin/` içinde `python-3.12.tar.gz` ve
`gdown-wheels/` bulunmalı. Wheelhouse yalnızca `requirements-lock.txt` değişirse yeniden
yapılır ve Drive'a yüklenir:

```bash
python3.12 -m pip wheel -w wheelhouse -r requirements-lock.txt \
    --extra-index-url https://download.pytorch.org/whl/cu129
tar -cf wheelhouse-py312-cu129.tar -C wheelhouse .
split -b 900M -d -a 2 wheelhouse-py312-cu129.tar wheelhouse-py312-cu129.tar.part-
```

Sonra parçaları Drive klasörüne yükle, yeni ID'leri `DRIVE_FILES.txt`'ye ve
tar'ın sha256'sını `SHA256SUMS.drive`'a yaz. Tek 4 GB'lık dosya halinde
yükleme, rclone'un ortak Google kotası yüzünden son adımda tekrar tekrar
başa dönmüştü (1 Kasım 2026).
