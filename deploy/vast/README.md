# vast.ai kurulumu: Türkçe dudak okuma fine-tune'u

Bu paket (`lr-v2-vast.tar`), boş bir Ubuntu vast.ai sunucusunda tam
fine-tune'u başlatmak için gereken her şeyi içerir: kod, kurulum ve test
betikleri, Drive linkleri ve sha256 değerleri. Büyük dosyalar (dataset,
Python kütüphaneleri, pretrained ağırlıklar) paketin içinde değildir;
`setup.sh` onları Google Drive'dan **gdown** ile indirir ve doğrular. Drive
klasörü linkle paylaşılmış olduğu için sunucuda Google hesabı ya da anahtar
gerekmez.

Akış:

```
laptop ──scp──> sunucu: lr-v2-vast.tar (137 MB)
sunucu: setup.sh   (Drive'dan indir, doğrula, aç, kur)        ~15-30 dk
sunucu: check.sh   (pytest, preflight, GPU bellek ölçümü)      ~15 dk
sunucu: train.sh full                                          günler
sunucu: finish.sh  (test seti, model paketi)                   ~30-60 dk
sunucu ──scp──> laptop: tr-vsr-model.tar     → sunucuyu kapat (Destroy)
```

## 0. Sunucuyu kiralarken

| Ayar | Değer | Neden |
|---|---|---|
| Image | `ubuntu:22.04` (veya 24.04), launch mode **SSH** | Kurulum kendi Python'unu getirir; başka bir şey gerekmez |
| GPU | Tek GPU, **en az 48 GB** (A6000, L40S, A100 80GB, H100) | 1.800 frame'lik batch 241M parametreli modelle; 24 GB kartta `FRAMES=900 ACCUM=2` gerekir |
| Max CUDA | **≥ 12.8** | Kütüphaneler PyTorch 2.13 + CUDA 12.9 ile derlendi |
| Disk | **≥ 100 GB** | Dataset 13+14 GB, kütüphaneler 11 GB, checkpoint'ler ~15 GB |
| RAM | ≥ 32 GB | Veri yükleyiciler ~0,5 GB/işçi; eğitim süreci 3-4 GB |
| CPU | ≥ 16 çekirdek | Videolar CPU'da açılır; az çekirdek GPU'yu bekletir |
| İnternet indirme | ≥ 500 Mbps | Drive'dan 18 GB indirilecek |

Birden fazla GPU'lu sunucu almayın: kod tek GPU kullanır.

## 1. Sunucuya ne indirilecek

`setup.sh` aşağıdaki dosyaları sırayla, Drive ID'leriyle indirir
(`DRIVE_FILES.txt`). Hepsi aynı Drive klasöründedir:
<https://drive.google.com/drive/folders/16tybAQsivB4GPGB2piDDpuoSJeWToBCS>

| # | Dosya | Boyut | Ne | Sunucuda nereye |
|---|---|---|---|---|
| 1 | [`tr-lrs-v3.tar.gz`](https://drive.google.com/file/d/15dwNvrNDL_YJbRYSsC7T_Vnk2Ng25jwC/view) | 12,9 GB | TR-LRS v3: 225.977 ağız kırpıntısı klip (96x96, 25 fps), manifest'ler, tokenizer | açılır: `lr-v2-codebase/dataset/`; tar silinir |
| 2 | [`vsr_trlrs2lrs3vox2avsp_base.pth`](https://drive.google.com/file/d/162axOfnnZxVfhpbm9XK4SKAo_l3Qdpf_/view) | 1,0 GB | Auto-AVSR (Ma ve ark.) görsel konuşma tanıma ağırlıkları, başlangıç noktası | `lr-v2-codebase/pretrained/` |
{{PART_ROWS}}

Wheelhouse 4,3 GB'lık tek bir tar (`wheelhouse-py312-cu129.tar`): 63
Python paketi (wheel), PyTorch 2.13.0+cu129 dahil. Drive'a yüklenirken
kesilmemesi için 5 parçaya bölündü (`split -b 900M`); `setup.sh`
parçaları `cat` ile birleştirir ve tam tar'ı doğrular.

sha256 (`SHA256SUMS.drive`):

```
143122937a84a16748a8b22b415d3d49b993c522f44a2b8842eb46d1dda55b8e  tr-lrs-v3.tar.gz
bc8f44ecf8d5218c23aa5ed3675350457758927b572d74675c7afa1325c06e03  wheelhouse-py312-cu129.tar
fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb  vsr_trlrs2lrs3vox2avsp_base.pth
```

Paketin kendi içinden gelenler (indirme yok):

| Dosya | Ne |
|---|---|
| `code.tar` | Kod, git commit'i `COMMIT` dosyasında. `lr-v2-codebase/` olarak açılır |
| `assets/` | `babble_noise.wav` (ses gürültüsü), `face_landmarker.task` (yüz noktaları; yalnızca model kullanımında gerekir) |
| `bin/python-3.12.tar.gz` | Taşınabilir Python 3.12.16 (python-build-standalone); sistem Python'u kullanılmaz |
| `bin/gdown-wheels/` | gdown 6.5.1 ve bağımlılıkları (12 wheel), internetsiz kurulur |
| `DRIVE_FILES.txt` | İndirilecek dosyaların Drive ID'leri |
| `requirements-lock.txt` | Kurulan paketlerin kesin sürümleri |
| `SHA256SUMS.drive` | Drive dosyalarının sha256 değerleri |
| `SHA256SUMS.bundle` | Paketin içindeki dosyaların sha256 değerleri |
| `MODEL_KULLANIMI.md` | Eğitilen modeli kullanma rehberi; `finish.sh` model paketine koyar |

Sunucuda son durum:

```
$WORK/                         /workspace (yoksa /root)
  lr-v2-codebase/              proje
    avsr/ scripts/ configs/ tests/ docs/ deploy/
    assets/
    dataset/tr-lrs-v3/         39 oturum klasörü, 451.954 dosya
    dataset/tr-lrs-v3-manifests/  train.tsv val.tsv test.tsv
    dataset/tr-lrs-v3-tokenizer/  unigram1000.model
    pretrained/vsr_trlrs2lrs3vox2avsp_base.pth
    .venv/                     Python 3.12 + kütüphaneler
    exp/vsr_finetune/          eğitim çıktısı (checkpoint'ler, metrikler)
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

`setup.sh` beş adımı sırayla yapar ve her adımı bitince işaretler. Bir
yerde durursa sorunu giderip yeniden çalıştırın; bitmiş adımlar atlanır,
yarım kalan indirme kaldığı yerden sürer.

1. **Kontrol:** `nvidia-smi`, sürücünün CUDA sürümü, boş disk
2. **Python ve gdown:** taşınabilir Python 3.12, gdown paketin içinden
3. **İndirme:** 1. bölümdeki dosyalar gdown ile; wheelhouse parçaları
   birleştirilir; `sha256sum -c SHA256SUMS.drive`
4. **Açma:** kod, assets, ağırlıklar; dataset açılır, oturum (39) ve dosya
   (451.954) sayısı kontrol edilir, dataset tar'ı silinir
5. **Kütüphaneler:** `.venv` → wheelhouse'tan internetsiz `pip install`;
   sonunda PyTorch'un GPU'da işlem yaptığı denenir

Son satır `KURULUM TAMAM` olmalı.

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
mkdir -p /workspace/lr-v2-codebase/pretrained && cd /workspace/lr-v2-codebase
tar -xf ~/lr-v2-vast/code.tar && cp -r ~/lr-v2-vast/assets .
cp /workspace/downloads/vsr_trlrs2lrs3vox2avsp_base.pth pretrained/
tar -xzf /workspace/downloads/tr-lrs-v3.tar.gz                  # dataset/
/workspace/python/bin/python3 -m venv .venv
mkdir -p /workspace/downloads/wh && tar -xf /workspace/downloads/wheelhouse-py312-cu129.tar -C /workspace/downloads/wh
.venv/bin/pip install --no-index --find-links /workspace/downloads/wh -r ~/lr-v2-vast/requirements-lock.txt
```

## 4. Testler

```bash
bash check.sh
```

| Test | Geçti sayılır | Süre |
|---|---|---|
| `pytest` | **0 failed**; ~205 passed. Bazıları `skipped`: referans kod, OpenCV ve ffmpeg yalnızca dataset yapılan makinede var | 5-10 dk |
| `scripts/preflight.py` | Son satır `READY TO TRAIN`. Her klip diskte mi, 500x3 klip açılıyor mu, 96x96 mı, tokenizer, pretrained ağırlıklar (762 tensör), disk, 20 adım gerçek eğitim | 3-5 dk |
| `scripts/benchmark_memory.py` | 1.800 frame satırında `peak GB`, GPU belleğinin ~%80'inden az | 2 dk |

Bellek sonucuna göre:

| 1.800 frame sığıyorsa | `bash train.sh full` |
|---|---|
| Yalnız 900 sığıyorsa | `FRAMES=900 ACCUM=2 bash train.sh full` |
| Yalnız 600 sığıyorsa | `FRAMES=600 ACCUM=3 bash train.sh full` |

## 5. Eğitim

```bash
bash train.sh full          # arka planda başlar; SSH kapansa da sürer
bash train.sh status        # epoch başına val/cer, GPU kullanımı, hata satırları
tail -f /workspace/lr-v2-codebase/logs/vsr_finetune.log
```

Tarif (`configs/finetune/vsr_base.yaml`): 75 epoch, yeni katmanlar lr 1e-3,
pretrained kısımlar lr 2e-5 (ilk epoch sabit tutulur), AdamW, 5 epoch
warm-up + cosine, 1.800 frame/batch, 0,1 CTC + 0,9 attention, eğitimde
Partition-Time Masking. 241M parametre. Ayrıntı: `docs/deployment.md` 8. bölüm.

Ne beklenmeli (laptopta 8.000 klipte ölçüldü, 1 Kasım 2026):

| Ne zaman | Beklenen |
|---|---|
| Log başı | `Loaded 762 tensors`, ardından iki öğrenme oranı satırı (1e-3 ve 2e-5) |
| 1. epoch sonu | `val/cer` 0,9'un altında (laptop denemesinde 3.675 güncellemeden sonra 0,68) |
| Sonraki epoch'lar | `val/cer` düşer |
| `val/cer` 5 epoch boyunca 0,9 üstünde | Eğitim öğrenmiyor: durdur (`pkill -f scripts/train.py`) |
| `status`'ta `amp_scale` sürekli düşüyor | 16-bit taşma: durdur, `bash train.sh resume train.precision=bf16-mixed` değil, baştan `EXP=vsr_bf16 bash train.sh full train.precision=bf16-mixed` |
| Logda `out of memory` / OOM | `FRAMES`'i küçült, `ACCUM`'u büyüt (çarpımları 1.800) |

Bir epoch'un süresini ilk epoch bitince `status` gösterir (adım sayısı ve
zaman). Toplam süre ≈ 75 x epoch süresi.

Bağlantı veya sunucu kesilirse:

```bash
bash train.sh resume        # last.ckpt'den, aynı ayarlarla sürer
```

İsteğe bağlı pilot (~2 saat): pretrained lr ölçeği 0,02 ve 0,1'in kısa
karşılaştırması. 0,02 küçük batch'lerle seçildi, 1.800 frame'de
doğrulanmadı.

```bash
bash train.sh pilot
```

## 6. Bitirme ve indirme

Eğitim bitince (`train.sh status` → `eğitim çalışmıyor`, 75 epoch):

```bash
bash finish.sh
```

1. Test seti (eğitimde hiç görülmemiş 2 oturum, 6.173 klip) beam search ile,
   son 10 checkpoint'in ortalamasıyla değerlendirilir:
   `exp/vsr_finetune/eval/test_video.json` (WER, CER) ve `.tsv` (her klibin
   referansı ve modelin çıktısı)
2. `tr-vsr-model/` klasörü yapılır ve kendi kendine yükleyip bir test
   klibini okuyarak denenir
3. `/workspace/tr-vsr-model.tar` (~1 GB) ve `.sha256` yazılır

Laptopta:

```bash
scp -P <port> 'root@<adres>:/workspace/tr-vsr-model.tar*' ~/Desktop/
cd ~/Desktop && sha256sum -c tr-vsr-model.tar.sha256 && tar -xf tr-vsr-model.tar
```

`sha256sum` `OK` dedikten sonra vast.ai panelinden sunucuyu **Destroy**
edin. Stop, diski saklamaya devam eder ve ücretlendirir.

Modeli kullanma: `tr-vsr-model/MODEL_KULLANIMI.md`.

`tr-vsr-model.tar` içinde:

| Dosya | Ne |
|---|---|
| `checkpoints/epoch-avg.ckpt` | Son 10 epoch'un ortalama ağırlıkları |
| `config.yaml` | Eğitim ayarları (model mimarisi buradan kurulur) |
| `tokenizer.model` | 1.000 birimlik Türkçe SentencePiece |
| `eval/test_video.json`, `.tsv` | Test sonuçları |
| `training_metrics.csv`, `train.log` | Eğitim boyunca metrikler ve log |
| `model_info.json` | Hangi checkpoint'ler ortalandı |
| `code.tar`, `assets/` | Modeli çalıştıran kod ve yüz bulucu modeli |
| `MODEL_KULLANIMI.md` | Kullanım rehberi |

## 7. Sorun giderme

| Belirti | Çözüm |
|---|---|
| gdown `Too many users have viewed or downloaded this file recently` | Drive'ın dosya başına indirme kotası; birkaç saat sonra `bash setup.sh` (kaldığı yerden sürer) |
| gdown `Cannot retrieve the public link` / `Access denied` | Drive klasörünün paylaşımı "Bağlantıya sahip olan herkes" olmalı |
| sha256 `FAILED` | `rm -f /workspace/.setup-done/download /workspace/downloads/<dosya>`, `bash setup.sh` |
| `PyTorch GPU'yu göremiyor` | Sürücü CUDA 12.8'den eski: başka sunucu seç |
| Disk doldu | `df -h /workspace`; `downloads/wheelhouse-py312-cu129.tar` silinebilir |

## Paketin yeniden yapılması (laptopta)

Kod değişirse, commit edip:

```bash
bash deploy/vast/make_bundle.sh ~/Desktop/vast-bundle-build ~/Desktop/lr-v2-vast.tar
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
