# Ham videolar: kaynaklar ve indirme

TR-LRS v3, bu belgedeki 39 oturumun ham kayıtlarından üretildi (`videos/`, toplam 180 GB). Ham dosyalar
2026-11-03'te diskten silindi. Üretilmiş dataset `dataset/tr-lrs-v3.tar.gz` içinde ve Drive'da duruyor.
Bu belge oturumların nereden geldiğini, nasıl indirildiğini ve yeniden indirilen bir dosyanın aynı olup olmadığının
nasıl kontrol edileceğini kaydeder.

**Kaynak:** 39 kaydın tamamı TBMM TV (Meclis TV) Genel Kurul yayınlarıdır ve hepsi bu yayınların
YouTube'daki yüklemelerinden indirildi. Bu yüzden dataset'e kaynak olarak TBMM TV gösterilmelidir; YouTube
linkleri sadece dosyaların nereden indirildiğini kaydeder.

Kaynak tablo `videos/sources.tsv` dosyasıdır; bir kopyası `docs/ham_videolar_sources.tsv` olarak burada duruyor. Bu dosya `python dataset-creation/corpus.py ingest` ile üretilir ve
tablolar ondan kopyalanmıştır.

## YouTube video ID'siyle adlandırılanlar: 20 oturum, 172.6 saat, 82.2 GB

Oturum adı `yt-<video id>` biçimindedir.

| Oturum | Link | Çözünürlük | Saat | Boyut (GB) | sha256 |
|---|---|---|---|---|---|
| `yt-1DIP6Ruvtgg` | https://www.youtube.com/watch?v=1DIP6Ruvtgg | 1920x1080 | 10.41 | 6.89 | `` |
| `yt-45SXDKapxh8` | https://www.youtube.com/watch?v=45SXDKapxh8 | 1920x1080 | 6.76 | 3.76 | `` |
| `yt-Fpah9dtIsFo` | https://www.youtube.com/watch?v=Fpah9dtIsFo | 1920x1080 | 7.19 | 4.19 | `` |
| `yt-HTCukfIyxoM` | https://www.youtube.com/watch?v=HTCukfIyxoM | 1920x1080 | 7.94 | 2.84 | `` |
| `yt-MZtO3ofW7eA` | https://www.youtube.com/watch?v=MZtO3ofW7eA | 1920x1080 | 8.63 | 4.69 | `` |
| `yt-NPJohShU_iw` | https://www.youtube.com/watch?v=NPJohShU_iw | 1920x1080 | 4.45 | 2.40 | `` |
| `yt-OPGJ5-ARIC4` | https://www.youtube.com/watch?v=OPGJ5-ARIC4 | 1920x1080 | 8.21 | 3.39 | `` |
| `yt-Pw91kFFU44o` | https://www.youtube.com/watch?v=Pw91kFFU44o | 1920x1080 | 7.2 | 4.27 | `` |
| `yt-QEdmcWbgxrg` | https://www.youtube.com/watch?v=QEdmcWbgxrg | 1920x1080 | 6.76 | 4.29 | `` |
| `yt-S_sCbEZeqCc` | https://www.youtube.com/watch?v=S_sCbEZeqCc | 1920x1080 | 11.33 | 4.27 | `` |
| `yt-UoHW7olV1LU` | https://www.youtube.com/watch?v=UoHW7olV1LU | 1920x1080 | 11.07 | 4.04 | `` |
| `yt-_3u6pzqTuoo` | https://www.youtube.com/watch?v=_3u6pzqTuoo | 1920x1080 | 11.34 | 6.00 | `` |
| `yt-_8zUPRDIwKM` | https://www.youtube.com/watch?v=_8zUPRDIwKM | 1920x1080 | 8.7 | 3.46 | `` |
| `yt-fP99G1PONhQ` | https://www.youtube.com/watch?v=fP99G1PONhQ | 1920x1080 | 6.48 | 2.47 | `` |
| `yt-hkcAzBtqACE` | https://www.youtube.com/watch?v=hkcAzBtqACE | 1920x1080 | 8.55 | 4.96 | `` |
| `yt-io-FZa_APv4` | https://www.youtube.com/watch?v=io-FZa_APv4 | 1920x1080 | 10.96 | 4.42 | `` |
| `yt-l419Asc8yqE` | https://www.youtube.com/watch?v=l419Asc8yqE | 1920x1080 | 11.43 | 3.88 | `` |
| `yt-r7XQxD_fBz4` | https://www.youtube.com/watch?v=r7XQxD_fBz4 | 1920x1080 | 10.43 | 3.80 | `` |
| `yt-sLfJGTUSnjw` | https://www.youtube.com/watch?v=sLfJGTUSnjw | 1920x1080 | 5.84 | 3.18 | `` |
| `yt-xZkvTV0J1PI` | https://www.youtube.com/watch?v=xZkvTV0J1PI | 1920x1080 | 8.87 | 4.99 | `` |

### İndirme kodu (son kullanılan)

Repodaki `dataset-creation/download_youtube_1080p.sh` ve `python dataset-creation/corpus.py ingest` aynı yt-dlp komutunu kullanır
(`avsr/corpus/sources.py: ytdlp_command`). Komut 1080p'ye kadar H.264 video ve AAC ses indirip tek bir mp4 dosyasında birleştirir:

```bash
.venv/bin/pip install -U yt-dlp      # 2026.08 veya daha yeni
# JavaScript runtime (node) gerekli. Yoksa YouTube her indirmeyi HTTP 403 ile reddeder.
yt-dlp --js-runtimes node --continue --no-overwrites --no-playlist \
  --retries 10 --fragment-retries 10 \
  -f 'bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=1080][ext=mp4]+ba/b[height<=1080]' \
  --merge-output-format mp4 -o 'videos/yt-%(id)s.%(ext)s' \
  "https://www.youtube.com/watch?v=<id>"
```

Tüm oturumları `sources.tsv` dosyasından yeniden indirmek için:

```bash
awk -F'\t' 'NR>1 && $2=="youtube" {print $3}' docs/ham_videolar_sources.tsv > yt_links.txt
```

Bu listeyi yukarıdaki komuta `-a yt_links.txt` ile ver.

İlk sürümde (`lr_codes/dlp`, 360p) kullanılan eski komut da `~/.bash_history` içinde kayıtlı:
`./yt-dlp_linux -a ./links.txt -f "best" -o "%(id)s.%(ext)s" --extractor-args "youtube:player_client=android" --download-archive downloaded.txt -N 8 --ignore-errors`.
Bu komut 360p indirir, v3 için kullanılmadı.

## Birleşim numarasıyla adlandırılanlar: 19 oturum, 159.9 saat, 98.1 GB

Bunlar da YouTube'dan indirilen TBMM TV Genel Kurul yayınlarıdır, ama dosyalar birleşime göre yeniden adlandırılmıştır. Oturum adı `<yasama yılı>-birlesim-<birleşim no>` biçimindedir, örneğin
`2024-2025-birlesim-72` = 2024-2025 yasama yılı, 72. birleşim. **YouTube linkleri kaydedilmemiş**
(`sources.tsv` dosyasındaki url sütunu boş). Videolar YouTube'da TBMM Genel Kurulu yüklemeleri arasında yasama yılı,
birleşim numarası ve tarihle bulunabilir; doğru video olduğunu süre ve boyut (aşağıdaki tablo) doğrular. Eski `video-N` adları ile bu adlar arasındaki eşleşme
`docs/audit_and_roadmap_2026-10-17.md` bölüm 6'da var.

| Oturum | Link | Çözünürlük | Saat | Boyut (GB) | sha256 |
|---|---|---|---|---|---|
| `2024-2025-birlesim-109` | — | 1920x1080 | 11.15 | 6.56 | `ebafd3d5c685844b5fecf76c0a65c7bad092f92f44584d4dd92cffba347338e7` |
| `2024-2025-birlesim-110` | — | 1920x1080 | 11.59 | 7.35 | `3633911def7de3f3f8744be06085d1b4a0a55f893beb93d83825c02762275d13` |
| `2024-2025-birlesim-72` | — | 1920x1080 | 9.2 | 5.00 | `25070e84a18a65e62a7e9a79e0390ccfc8035dea920e978b1399f02247befff9` |
| `2024-2025-birlesim-73` | — | 1920x1080 | 8.69 | 5.70 | `5ba3594bb7d8e5f5eefedba89cec4969aead122f3eb9ff69220574e3f8483d7a` |
| `2024-2025-birlesim-74` | — | 1920x1080 | 8.23 | 4.97 | `259b5b2180270b816f424e00c9a811292367bbae4b7aed37dbb1c6475f2e3076` |
| `2024-2025-birlesim-76` | — | 1920x1080 | 7.65 | 8.24 | `9ccaa0d3f8f4302e11d3e313c6ea91c26ef72e3c9c9959d81e911c9db878f596` |
| `2024-2025-birlesim-80` | — | 1920x1080 | 4.38 | 2.75 | `785478a7905a14f3e6266cbcc40bbe242dcbef8cb8871c78ef2f95a717afc5f2` |
| `2024-2025-birlesim-82` | — | 1920x1080 | 9.32 | 5.84 | `0d3bf7ca0725bd987aa1676a6418a14ed8c950af1dac1c971f76049ec6d26837` |
| `2024-2025-birlesim-84` | — | 1920x1080 | 8.84 | 5.48 | `9237a151aa1d3b5e4654710d92c94f53a5a74cb395c07d6c42fa050215c96d0e` |
| `2024-2025-birlesim-94` | — | 1920x1080 | 11.31 | 7.88 | `038f751c1130e2de0c97bb43f7d4146b700c4db7f096296a1afe796ef6e8dc51` |
| `2025-2026-birlesim-1` | — | 1920x1080 | 8.47 | 4.99 | `1138eead01a3c128e72e3f2811acd7b41cc1e8781ad00739758bce55c880db24` |
| `2025-2026-birlesim-10` | — | 1920x1080 | 10.41 | 6.94 | `2d0465f671ebd79cfbe1ce5ee844841b6e21b9fc5a29993dc13e5d32b339fe69` |
| `2025-2026-birlesim-12` | — | 1920x1080 | 7.18 | 4.44 | `edea9289fae0bccc0dc28673eb1693866d070cc140d4f56205375b776ccffa3b` |
| `2025-2026-birlesim-2` | — | 1920x1080 | 6.68 | 3.62 | `491cbe0ead006a1cd913bd89bc7bc7826245b8251eba6ba41913cc9dadefb031` |
| `2025-2026-birlesim-3` | — | 1920x1080 | 4.81 | 3.21 | `79ebc2657e1c4b743633650dc3e25853a3b3eecb8f86f393c45db4e3c2d2ab67` |
| `2025-2026-birlesim-5` | — | 1920x1080 | 8.07 | 2.87 | `279d41b2777cc10eb1a446999e544b9c2c3b317eaf85681a4f717f8bbb2b952e` |
| `2025-2026-birlesim-7` | — | 1920x1080 | 7.9 | 4.49 | `8ac893a5af3821fa936e6bdd5e9ea17bfe5e92745d90cd13cce2f66ea779b2ac` |
| `2025-2026-birlesim-8` | — | 1920x1080 | 8.03 | 5.07 | `cf23c58b5d48b591bfaf042be91bbc9d4420de292306695abb9254758507b674` |
| `2025-2026-birlesim-9` | — | 1920x1080 | 7.96 | 2.73 | `ef7d5a301071d7c85b933e108627667c9f2cb604fb4d2228a26fee05752e01fb` |

## Yeniden indirilen bir dosyayı doğrulama

```bash
sha256sum videos/<oturum>.mp4      # tablodaki sha256 ile aynı olmalı
python dataset-creation/corpus.py ingest    # sources.tsv dosyasını yeniden yazar, çözünürlük ve süreyi kontrol eder
```

YouTube ve TBMM aynı dosyayı yeniden kodlayarak sunabilir. Böyle bir durumda sha256 tutmaz. Çözünürlük, fps ve süre
tutuyorsa içerik aynıdır, ancak klip sınırları `dataset/tr-lrs-v3.clips.tsv` dosyasındaki zamanlara göre yeniden kesilmelidir.
