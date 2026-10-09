# WiLoR -> G1 canli kullanim rehberi

Kamera goruntusundeki ellerinizi (WiLoR) takip eder; ayni anda yuzen eller ve Unitree G1 (xr_teleoperate IK + Dex3
parmaklar) MuJoCo'da size eslik eder. Hepsi bu makinede (Raider) calisir.

## 1. Baslatma

Iki terminal, repo kokunden (`/home/aksoy-msi/code/wilor_faruk`):

```bash
# Terminal 1 - el takip sunucusu (wilor-venv, GPU). ZED 2i icin:
data/wilor-venv/bin/python xr_camera_teleop/api/server.py --source zed

# Terminal 2 - canli arayuz (xr-env)
data/xr-env/bin/python xr_camera_teleop/ui/live_ui.py
```

Tarayicida acin: **http://127.0.0.1:8780/**  (baska bir makineden: `live_ui.py --host 0.0.0.0` ve sunucu icin
`server.py --host 0.0.0.0`, sonra `http://<raider-ip>:8780/`).

Sayfada uc panel vardir: sol WiLoR goruntusu (isaretler, bilek eksenleri, sol/sag etiketi), orta yuzen eller,
sag G1. Ustte dugmeler, altta durum satiri (kalibrasyon, el durumu, egim, fps, gecikme) bulunur.
Durum satirinda `self-collision: ON` gorunur: G1 IK'si kol-govde ve kol-kol carpismasini engeller (eller/bilekler
birbirine degebilir), kollar garip bir konumda takilirsa kendiliginden duzgun konfigurasyona yavasca gecer
(`xr_camera_teleop/teleop/safe_arm_ik.py`, ayrintilar RUN.md). **Reset** (R) kollari eklem uzayinda baslangic durusuna (q=0)
yavasca geri getirir; sonra yeniden kalibrasyon. Kapatmak icin `live_ui.py --no-self-collision`.
Sunucuda `--source` yerine bir video dosyasi (`--source kayit.mkv`) de verilebilir.

## 2. Varsayimlar

- Kamera herhangi bir yonden bakabilir: bas/gogus (egocentric), omuz ustu veya masanin karsisindan size dogru.
  Kollarin goruntunun neresinden geldigi onemli degil. Kalibrasyon hem konum farkini siler hem de
  **ileri yonu** (yaw) otomatik bulur: kalibrasyon anindaki iki elin bilek->orta parmak yonunun ortalamasi
  "ileri" kabul edilir. Bu yuzden kalibrasyonda **eller onunuzde, parmaklar ileri bakacak sekilde** durmali
  (masaya rahatca koymak yeterli). Bulunan yon durum satirinda `yaw` olarak gorunur (egocentric ~0, karsidan ~180).
- Kamera egimi (pitch/roll) ZED IMU ile **otomatik** duzeltilir (`cam_R_level`); egim icin elle ayar gerekmez.
  IMU yalnizca pitch ve roll verir; yaw kalibrasyonda ellerden bulunur (yukari bakin).
  Kamera IMU'suz ise sayfadaki "Cam pitch" kutusundan elle derece verin (+ = asagi bakiyor).
- Kameraya 0.9 m'den uzak eller yok sayilir (operatorun eli degil sayilir).

## 3. Kalibrasyon

1. Rahat oturun. Iki elinizi onunuzde, **ikisi de goruntunun icinde** (kenardan en az ~40 piksel iceride) ve
   kameraya 0.9 m'den yakin tutun.
2. Elleri yaklasik **1 saniye sabit** tutun. Durum satiri "waiting: hold both hands still 1 s" iken "CALIBRATED" olur; orta panelde
   "calibrated" yazar ve G1 elleri robotun dinlenme konumundan baslar.
3. Sonra ellerinizi hareket ettirin: G1 hareketinizi izler. Iki el ortak bir referans noktasi paylasir; eller
   birbirine yaklasirsa G1 elleri de yaklasir.
4. Isterseniz `C` tusu veya "Calibrate now" ile hemen kalibre edin (iki el goruntude gorulunce uygulanir).

## 4. Kontroller

| Kontrol | Tus | Ne yapar |
| --- | --- | --- |
| Calibrate now | `C` | Mevcut el konumunu robotun dinlenme konumuyla eslestirir |
| Reset | `R` | Kalibrasyonu siler; eller sabit durunca yeniden otomatik kalibre olur |
| Pause / engage | `Bosluk` | Duraklatir: G1 son pozunda kalir. Tekrar basinca elleriniz o anki robot pozuna "yapistirilir" (ziplama olmaz) |
| k kaydirici (0.3 - 1.5) | | Konum kazanci: robot el hareketi = k x insan el hareketi. 0.8 varsayilan; kucuk k daha guvenli/dar, buyuk k daha genis hareket |
| Cam pitch | | `auto (IMU)` varsayilan. `manual` secip derece girerseniz IMU yok sayilir |

Bir el 1 saniyeden uzun kaybolursa o kol yavasca dinlenme konumuna doner; el geri gelince ziplamadan devam eder.

## 5. Kayit ve sonradan render

```bash
# Sunucuyu kayitla baslatin (ham kamera + WiLoR mesajlari)
data/wilor-venv/bin/python xr_camera_teleop/api/server.py --source zed \
    --record data/sessions/hands_ORNEK.jsonl \
    --record-video data/sessions/cam_ORNEK
# Ctrl+C ile durdurun. cam_ORNEK.mkv/.csv/.json ve hands_ORNEK.jsonl olusur (ORNEK = ayni zaman damgasi olmali).

# Uc panelli mp4 (sol WiLoR, orta yuzen eller, sag G1)
data/xr-env/bin/python xr_camera_teleop/tools/render_session.py --session data/sessions/ORNEK [--k 0.8] [--start S --duration S]
```

Cikti `data/sessions/render_ORNEK.mp4`. Mesajlar `cam_R_level` (IMU) iceriyorsa egim otomatik kullanilir;
eski kayitlarda `--cam-pitch-deg 43` gibi elle verilir.
Kayitli el verisini canli arayuze tekrar oynatmak icin:
`data/xr-env/bin/python xr_camera_teleop/api/replay.py --file hands_ORNEK.jsonl --loop --realtime` (sunucu yerine).

## 6. Sorun giderme

- **Port 5590 kullanimda** (`Address already in use`): sunucu veya `replay.py` zaten calisiyor. `pgrep -af "api/server.py|replay.py"`
  ile bulup durdurun; ayni anda yalnizca biri 5590'a yayin yapar. Arayuz icin 8780, sunucu HTTP icin 8770 kullanilir.
- **El gorunmuyor / durum "stale" veya "never"**: eller kamerada gorunuyor mu (sol panel)? 0.9 m'den uzaktaysa yok sayilir.
  Isik yeterli mi, el goruntunun kenarinda kesilmis mi? `curl localhost:8770/health` `hands` alanini gosterir.
  Hic goruntu yoksa ZED baska bir surec tarafindan kullaniliyor olabilir.
- **Kalibre olmuyor**: iki el de iceride, yakin ve 1 s sabit olmali; kalibrasyon sirasinda eller 3 cm'den fazla oynamamali.
  Elle `C` ile deneyin.
- **Sag/sol ters / G1 ters yone gidiyor**: kalibrasyonda parmaklar ileri bakmiyordu, `yaw` yanlis bulundu. Elleri onunuze,
  parmaklar ileri koyup R ardindan C (veya 1 s sabit tutun). Sol panelde
  yesil = sizin sol eliniz, mavi = sag eliniz olmali (etiket yaziyor).
- **G1 garip/yanlis yone gidiyor**: egim yanlis olabilir. Durum satirindaki `tilt` kaynagi IMU olmali; degilse ve
  kamera egikse manuel pitch girin. Sonra `R` ile yeniden kalibre edin.
- **Arayuz yavas / takiliyor**: durum satirindaki fps'e bakin. MuJoCo panelleri en iyi gayret ile 30 fps'e calisir;
  `live_ui.py --fps 20` ile dusurun, baska agir islemleri kapatin. Sol panel sunucudan dogrudan akar; sayfada
  birden fazla sekme acik tutmayin.
