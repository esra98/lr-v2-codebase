# vast.ai kurulumu: Türkçe dudak okuma fine-tune'u

Bu paket (`lr-v2-vast.tar`), boş bir Ubuntu vast.ai sunucusunda tam
fine-tune'u başlatmak için gereken her şeyi içerir: kod, kurulum ve test
betikleri, Drive bağlantıları ve sha256 değerleri. Büyük dosyalar (dataset,
Python kütüphaneleri, pretrained ağırlıklar) paketin içinde değildir;
`setup.sh` onları Google Drive'dan indirir ve doğrular.

Akış:

```
laptop ──scp──> sunucu: lr-v2-vast.tar + rclone.conf
sunucu: setup.sh   (Drive'dan indir, doğrula, aç, kur)        ~15-30 dk
sunucu: check.sh   (pytest, preflight, GPU bellek ölçümü)      ~15 dk
sunucu: train.sh full                                          günler
sunucu: finish.sh  (test seti, model paketi)                   ~30 dk
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

`setup.sh` aşağıdaki dosyaları tek tek indirir. Hepsi aynı Drive klasöründedir:
<https://drive.google.com/drive/folders/16tybAQsivB4GPGB2piDDpuoSJeWToBCS>

| # | Dosya | Boyut | Ne | Sunucuda nereye | sha256 |
|---|---|---|---|---|---|
| 1 | [`tr-lrs-v3.tar.gz`](https://drive.google.com/file/d/15dwNvrNDL_YJbRYSsC7T_Vnk2Ng25jwC/view) | 12,9 GB | TR-LRS v3: 225.977 ağız kırpıntısı klip (96x96, 25 fps), manifest'ler, tokenizer | açılır: `lr-v2-codebase/dataset/`; tar silinir | `143122937a84a16748a8b22b415d3d49b993c522f44a2b8842eb46d1dda55b8e` |
| 2 | [`wheelhouse-py312-cu129.tar`](WHEELHOUSE_LINK) | 4,3 GB | 63 Python paketi (wheel), PyTorch 2.13.0+cu129 dahil; internetsiz kurulum | `.venv` içine kurulur; tar kalır | `bc8f44ecf8d5218c23aa5ed3675350457758927b572d74675c7afa1325c06e03` |
| 3 | [`vsr_trlrs2lrs3vox2avsp_base.pth`](PRETRAINED_LINK) | 1,0 GB | Auto-AVSR (Ma ve ark.) görsel konuşma tanıma ağırlıkları, başlangıç noktası | `lr-v2-codebase/pretrained/` | `fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb` |

Paketin kendi içinden gelenler (indirme yok):

| Dosya | Ne |
|---|---|
| `code.tar` | Kod, git commit'i `COMMIT` dosyasında. `lr-v2-codebase/` olarak açılır |
| `assets/` | `babble_noise.wav` (ses gürültüsü), `face_landmarker.task` (yüz noktaları; yalnızca model kullanımında gerekir) |
| `bin/python-3.12.tar.gz` | Taşınabilir Python 3.12.16 (python-build-standalone); sistem Python'u kullanılmaz |
| `bin/rclone` | rclone 1.75.2, Drive'dan indirmek için |
| `requirements-lock.txt` | Kurulan paketlerin kesin sürümleri |
| `SHA256SUMS.drive` | Yukarıdaki 3 Drive dosyasının sha256 değerleri |
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
  downloads/                   wheelhouse tar'ı, pretrained'in kopyası
  bin/rclone
```

## 2. Laptoptan sunucuya yükleme

vast.ai panelindeki "Connect" düğmesi SSH komutunu verir:
`ssh -p <port> root@<adres>`. Aynı port ve adresle:

```bash
cd ~/Desktop
sha256sum -c lr-v2-vast.tar.sha256
scp -P <port> lr-v2-vast.tar ~/.config/rclone/rclone.conf root@<adres>:/root/
```

`rclone.conf`, Drive hesabına (`gdrive` bağlantısı) erişim anahtarını
taşır. Paketin içine bilerek konmadı. Sunucu kapatılınca (Destroy) silinir.

## 3. Kurulum

```bash
ssh -p <port> root@<adres>
cd /root && tar -xf lr-v2-vast.tar && mv rclone.conf lr-v2-vast/ && cd lr-v2-vast
bash setup.sh
```

`setup.sh` dört adımı sırayla yapar ve her adımı bitince işaretler. Bir
yerde durursa sorunu giderip yeniden çalıştırın; bitmiş adımlar atlanır.

1. **Kontrol:** `nvidia-smi`, sürücünün CUDA sürümü, boş disk, rclone ayarı
2. **İndirme:** 1. bölümdeki üç dosya, ardından `sha256sum -c SHA256SUMS.drive`
3. **Açma:** kod, assets, ağırlıklar; dataset açılır, oturum (39) ve dosya
   (451.954) sayısı kontrol edilir, dataset tar'ı silinir
4. **Python:** taşınabilir Python 3.12 → `.venv` → wheelhouse'tan internetsiz
   `pip install`; sonunda PyTorch'un GPU'da işlem yaptığı denenir

Son satır `KURULUM TAMAM` olmalı.

Elle yapmak isterseniz eşdeğer komutlar:

```bash
export PATH=/workspace/bin:$PATH; mkdir -p /workspace/{bin,downloads}
install -m 755 bin/rclone /workspace/bin/
mkdir -p ~/.config/rclone && cp rclone.conf ~/.config/rclone/
for f in tr-lrs-v3.tar.gz wheelhouse-py312-cu129.tar vsr_trlrs2lrs3vox2avsp_base.pth; do
  rclone copy gdrive: /workspace/downloads --bind 0.0.0.0 \
      --drive-root-folder-id 16tybAQsivB4GPGB2piDDpuoSJeWToBCS --include "/$f" -P
done
(cd /workspace/downloads && sha256sum -c ~/lr-v2-vast/SHA256SUMS.drive)
mkdir -p /workspace/lr-v2-codebase/pretrained && cd /workspace/lr-v2-codebase
tar -xf ~/lr-v2-vast/code.tar && cp -r ~/lr-v2-vast/assets .
cp /workspace/downloads/vsr_trlrs2lrs3vox2avsp_base.pth pretrained/
tar -xzf /workspace/downloads/tr-lrs-v3.tar.gz          # dataset/
tar -xzf ~/lr-v2-vast/bin/python-3.12.tar.gz -C /workspace
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
| `rclone'da 'gdrive:' adında bir Drive bağlantısı yok` | 2. bölümdeki `scp` ile `rclone.conf`'u gönder, `lr-v2-vast/` içine koy |
| rclone `token expired` / `invalid_grant` | Laptopta `rclone config reconnect gdrive:`, sonra conf'u tekrar gönder |
| `network is unreachable` (IPv6) | İndirme `--bind 0.0.0.0` ile IPv4'e zorlanır; sürerse sunucuyu değiştir |
| sha256 `FAILED` | `rm -f /workspace/.setup-done/download /workspace/downloads/<dosya>`, `bash setup.sh` |
| `PyTorch GPU'yu göremiyor` | Sürücü CUDA 12.8'den eski: başka sunucu seç |
| Drive günlük indirme kotası (`downloadQuotaExceeded`) | Birkaç saat bekle ya da dosyayı Drive'da kopyalayıp ID'yi değiştir |
| Disk doldu | `df -h /workspace`; `downloads/wheelhouse-py312-cu129.tar` silinebilir |

## Paketin yeniden yapılması (laptopta)

Kod değişirse, commit edip:

```bash
bash deploy/vast/make_bundle.sh ~/Desktop/vast-bundle-build ~/Desktop/lr-v2-vast.tar
```

`~/Desktop/vast-bundle-build/bin/` içinde `rclone` ve `python-3.12.tar.gz`
bulunmalı. Wheelhouse yalnızca `requirements-lock.txt` değişirse yeniden
yapılır ve Drive'a yüklenir:

```bash
python3.12 -m pip wheel -w wheelhouse -r requirements-lock.txt \
    --extra-index-url https://download.pytorch.org/whl/cu129
```
