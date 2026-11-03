# Remote run: vast.ai'de eğitimi başlatma (kısa yol)

Ayrıntılar: `KURULUM.md`. Model kullanımı: `MODEL_KULLANIMI.md`.

## 1. Sunucuyu kirala

| Ayar | Değer |
|---|---|
| GPU | Tek GPU, en az 48 GB (A6000, L40S, A100 80GB, H100) |
| Max CUDA | ≥ 12.8 |
| Disk | ≥ 100 GB |
| RAM / CPU | ≥ 32 GB / ≥ 16 çekirdek |
| Image | `ubuntu:22.04`, launch mode SSH |

## 2. Paketi gönder (laptopta)

vast.ai panelindeki "Connect" SSH komutunu verir: `ssh -p <port> root@<adres>`.

```bash
cd ~/Desktop
sha256sum -c lr-v2-vast.tar.sha256
scp -P <port> lr-v2-vast.tar root@<adres>:/root/
```

## 3. Kur, test et, başlat (sunucuda)

```bash
ssh -p <port> root@<adres>
cd /root && tar -xf lr-v2-vast.tar && cd lr-v2-vast
bash setup.sh        # Drive'dan gdown ile indirir, sha256 doğrular, kurar   (~16 dk)
bash check.sh        # pytest + preflight + GPU bellek ölçümü                  (~15 dk)
bash train.sh full   # tam fine-tune, arka planda; SSH kapansa da sürer
```

- `setup.sh`'nin son satırı `KURULUM TAMAM` olmalı. Yarıda kalırsa tekrar
  çalıştır; bitmiş adımlar atlanır, indirme kaldığı yerden sürer.
- `check.sh` sonunda hangi komutla başlatacağını yazar. GPU 1.800 frame'lik
  batch'i taşımıyorsa örneğin: `FRAMES=900 ACCUM=2 bash train.sh full`.

## 4. İzle

```bash
bash train.sh status     # epoch başına val/cer, val/wer, GPU, hata satırları
tail -f /workspace/tr-lipreading/logs/vsr_finetune.log
```

| Ne zaman | Beklenen |
|---|---|
| 1. epoch sonu | `val/cer` 0,9'un altında (laptop denemesinde 0,68) |
| Sonraki epoch'lar | `val/cer` düşer |
| `val/cer` 5 epoch boyunca 0,9 üstünde | Öğrenmiyor: `pkill -f "^python training/train.py"` |

Bağlantı ya da sunucu kesilirse: `bash train.sh resume`.

## 5. Bitir, indir, kapat

Eğitim bitince (`train.sh status` → `eğitim çalışmıyor`):

```bash
bash finish.sh           # test seti + model paketi: /workspace/tr-vsr-model.tar
```

Laptopta:

```bash
scp -P <port> 'root@<adres>:/workspace/tr-vsr-model.tar*' ~/Desktop/
cd ~/Desktop && sha256sum -c tr-vsr-model.tar.sha256 && tar -xf tr-vsr-model.tar
```

`OK` gördükten sonra vast.ai panelinden sunucuyu **Destroy** et (Stop diski
saklar ve ücretlendirmeye devam eder). Modeli kullanmak için:
`tr-vsr-model/MODEL_KULLANIMI.md`.

`setup.sh`, `check.sh` ya da ilk epoch'ta beklenmedik bir şey olursa çıktıyı
olduğu gibi sakla; sorun gidermede ilk bakılacak yer `KURULUM.md` 7. bölüm.
