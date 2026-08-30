# Proctoring AI — Layihə Konteksti

## 1. Layihə haqqında

Universitet bitirmə layihəsi: sinif otağının video görüntüsünü analiz edərək imtahanda köçürmə hallarını (telefon istifadəsi, kağız ötürmə, sağa-sola baxma, üzərinə köçürmə) aşkarlayan AI sistemi.

- Kamera: küncdən quraşdırılmış, geniş bucaqlı, uzaq məsafəli, eyni anda 10+ tələbəni əhatə edir.
- Ayrıca, hər köçürmə tipi üçün yaxın plan (close-up) referans videolar mövcuddur.
- Etiketli klip data (phone use, looking around, paper passing, copying) **ground-truth validation üçündür, training üçün DEYİL**.

## 2. Faktiki qovluq strukturu

`proctoring/` wrapper YOXDUR — hər şey kök qovluqda `core/` altındadır.

```
core/
├── tracker.py          ✅ tamamlanıb
├── detector.py          ✅ tamamlanıb (qismən — aşağıya bax)
├── pose_utils.py         ✅ tamamlanıb (= planlaşdırılan pose_estimator.py)
├── head_pose_yolo.py    ✅ tamamlanıb
└── hand_proximity.py     ✅ tamamlanıb (= planlaşdırılan interaction_analysis.py; sadələşdirilib — bax bölmə 3)

reporting/
└── visualizer.py         ✅ tamamlanıb (draw_hand_proximity, draw_alert_labels)

tests/
├── test_hand_proximity.py ✅ tamamlanıb (run_hand_proximity_test)
├── test_tracker.py         ✅ tamamlanıb (run_tracker_on_video, tracker.py-dən köçürülüb)
├── test_detector.py        ✅ tamamlanıb (run_detector_test, detector.py-dən köçürülüb)
└── test_head_pose_yolo.py  ✅ tamamlanıb (run_yolo_pose_test, head_pose_yolo.py-dən köçürülüb)

validation/                (YAZILMAYACAQ — sadələşdirildi, bax bölmə 5)
main.py                    ✅ tamamlanıb (sadə, skorsuz versiya — bax bölmə 5, "Mərhələ 5/6 sadələşdirmə")
```

**Qeyd:** `analysis/` qovluğu və onun içindəki `risk_scorer.py`/`config.py`, eləcə də `reporting/report_generator.py` **silinib** (Mərhələ 5/6 sadələşdirmə qərarı, bax bölmə 5). Tarixçə/motivasiya üçün bax həmin bölmə — kod GitHub tarixçəsində qalır.

## 3. Tamamlanmış modullar (faktiki, kod analizinə əsasən)

**`core/tracker.py`** — YOLO (`yolo26m.pt`) + ByteTrack, insan aşkarlama/izləmə.
- `TrackedBox` (dataclass: `track_id, x1,y1,x2,y2,conf`) — demək olar bütün digər fayllarda import olunur. **İnterfeysi dəyişməyib** (bax bölmə 9).
- `PersonTracker.update_frame()`, `extract_tracks()`, `draw_tracks()`.
- I/O: giriş `np.ndarray` frame → çıxış `list[TrackedBox]`.
- Heç nədən asılı deyil (kök modul).
- ✅ **Köçürüldü:** `run_tracker_on_video()` (+ video `__main__` çağırışı) → `tests/test_tracker.py`-ə (bölmə 6-dakı prinsipə uyğun, core yalnız aşkarlama məntiqini saxlayır).
- ✅ **`supervision` ilə sadələşdirildi (bax bölmə 9):** `extract_tracks()` daxili parsing-i `sv.Detections.from_ultralytics()` ilə edir (çıxış yenə `list[TrackedBox]`). `draw_tracks()`-dəki `_ID_COLOR_CACHE`/`_color_for_id` hand-rolled rəng-keşi silinib, `sv.BoxAnnotator(color_lookup=TRACK)` ilə əvəzlənib (yalnız çəkmə anında müvəqqəti `sv.Detections` qurulur).

**`core/detector.py`** — Telefon aşkarlanması + ən yaxın şəxsə bağlama (owner association).
- `PhoneBox` (owner_id sahəsi ilə), `detect_phones()`, `associate_phones()`.
- I/O: giriş frame + `list[TrackedBox]` → çıxış `list[PhoneBox]` (owner_id təyin olunmuş).
- `tracker.py`-dan `TrackedBox` import edir. `pose_utils`/`head_pose_yolo` ilə əlaqəsi yoxdur — ayrı pipeline, öz YOLO çağırışını edir.
- ⚠️ **Naqislik:** yalnız tam-kadr detection var. Crop-based fallback (aşağıda "açıq problemlər"də qeyd olunan hibrid yanaşma) hələ implementasiya olunmayıb.
- ✅ **`supervision` ilə sadələşdirildi (bax bölmə 9):** `detect_phones()` daxili parsing-i `sv.Detections.from_ultralytics(results[0])` ilə edir — manual `.xyxy`/`.conf` çıxarma VƏ `if len(results[0].boxes) == 0: return []` yoxlaması (boş `zip()` avtomatik `[]` verdiyi üçün) silindi. `PhoneBox` (owner_id daxil) və `associate_phones()` TAM TOXUNULMAZ qalıb.
- ✅ **Köçürüldü:** tam-pipeline `__main__` demo bloku `run_detector_test()` funksiyasına çevrilib `tests/test_detector.py`-ə köçürülüb; köhnəlmiş `models\yolov8m.pt` yolu `yolo26m.pt` ilə düzəldildi (layihədə hər yerdə istifadə olunan model).

**`core/pose_utils.py`** — Ortaq YOLO-Pose abstraksiyası (bir dəfə işləyir, keypoint-ləri paylayır).
- Keypoint indeksləri (`NOSE=0…RIGHT_WRIST=10`), `PoseConfig`, `is_visible()`, `crop_person_for_pose()`, `get_pose_keypoints()`, `get_keypoints_for_person()`, `get_keypoints_for_all()`.
- I/O: giriş pose model + frame + `TrackedBox` → çıxış `(17,3)` numpy array (COCO keypoint, frame-koordinatında) və ya `dict[track_id, keypoints]`.
- `tracker.py`-dan `TrackedBox` import edir. `head_pose_yolo.py` və `hand_proximity.py` hər ikisi bunu istehlak edir — dublikat inference-dən qaçır (planın əsas məqsədi budur, artıq işləyir).

**`core/head_pose_yolo.py`** — Keypoint-lərdən yaw bucağı → baxış istiqaməti.
- `GazeDirection` (Enum: FORWARD/SIDE/AWAY/FACE_NOT_VISIBLE), `HeadPoseResult`, `GazeConfig`, `estimate_yaw_from_keypoints()`, `classify_direction()`, `estimate_head_pose_yolo()`, `draw_head_pose_yolo()`.
- I/O: giriş `pose_utils`-dan keypoint (və ya öz-özü hesablaya bilir) → çıxış `HeadPoseResult(track_id, yaw, direction)`.
- ⚠️ **Tarixçə:** Mərhələ 5-də `YawEvent`/`YawEventTracker` əlavə olunmuşdu (risk_scorer üçün diskret event), sonra Mərhələ 5/6 sadələşdirmə qərarı ilə silindi (bax bölmə 5). Fayl indi öz ilkin, sadə formasındadır — `HeadPoseResult` heç vaxt dəyişmədi.
- ✅ **Köçürüldü:** `run_yolo_pose_test()` (+ video `__main__` çağırışı) → `tests/test_head_pose_yolo.py`-ə (bölmə 6-dakı prinsipə uyğun).

**`core/hand_proximity.py`** — Bilək (wrist) yaxınlığı ilə kağız ötürmə siqnalı. Yalnız aşkarlama məntiqini saxlayır (bölmə 6-dakı prinsipə uyğun sadələşdirilib).
- `HandProximityConfig`, `PairProximityResult`, `HandProximityTracker` (class, ardıcıl frame state saxlayır — `_consecutive_counts` dict).
- I/O: giriş `dict[track_id, keypoints]` (`pose_utils.get_keypoints_for_all()`-dan) → çıxış hər frame üçün `list[PairProximityResult(track_id_1, track_id_2, distance, is_sustained)]` — **per-frame state**, `required_consecutive_frames=10` istifadə olunur (FPS-dən asılı, saniyəyə çevrilməyib).
- ✅ **Sadələşdirmə tamamlanıb:** `draw_hand_proximity` → `reporting/visualizer.py`-ə, `run_hand_proximity_test` (+ video `__main__` çağırışı) → `tests/test_hand_proximity.py`-ə köçürülüb. `PairProximityResult.hand_1/hand_2` və `is_close` sahələri (istifadəçi təsdiqi ilə) silindi.
- ⚠️ **Tarixçə:** Mərhələ 5-də `HandProximityEvent`/`HandProximityEventTracker` əlavə olunmuşdu, sonra Mərhələ 5/6 sadələşdirmə qərarı ilə silindi (bax bölmə 5). Əsas tracking məntiqi (`update()`) heç vaxt dəyişmədi — YALNIZ `main.py`-dakı müvəqqəti debug rejimi üçün əlavə olunmuş `get_consecutive_count()` getter-i əlavə olunub, sonra silinib (bax `main.py` bölməsi).

**`reporting/visualizer.py`** — Core modulların nəticələrinin frame üzərinə çəkilməsi.
- `draw_hand_proximity()` (`hand_proximity.py`-dən köçürülüb) — I/O: giriş frame + `dict[track_id, TrackedBox]` + `list[PairProximityResult]` → çıxış annotasiya olunmuş frame (in-place + qaytarır).
- ✅ **`draw_alert_labels(frame, track_box, labels: list[str])`** — bbox üzərində hər label-i ayrı sətirdə yazır, sadə sabit rəng (telefon="telefon" sözünü ehtiva edən label = qırmızı, digərləri = sarı). Skor/kateqoriya YOXDUR.
- `core/hand_proximity.py`-dən `PairProximityResult`, `core/tracker.py`-dən `TrackedBox` import edir (`sys.path`-ə `core/` əlavə olunur — layihədə paket strukturu yoxdur, bax bölmə 7).
- ⚠️ **Tarixçə:** Mərhələ 6-da `draw_risk_label()` (risk_scorer nəticəsi üçün rəng-kodlu label + blink effekt) əlavə olunmuşdu, sonra Mərhələ 5/6 sadələşdirmə qərarı ilə silindi, `draw_alert_labels()` ilə əvəzləndi (bax bölmə 5).

**`tests/test_hand_proximity.py`** — `hand_proximity.py` üçün tam-pipeline test/demo skripti (`run_hand_proximity_test()`, `hand_proximity.py`-dən köçürülüb).
- Real video açır, `cv2.imshow` ilə göstərir — avtomatlaşdırılmış test deyil, əl ilə yoxlama üçündür.
- `core/` və `reporting/`-dən import edir (`sys.path` genişləndirməsi ilə).

**`main.py`** (kök qovluqda) — bütün pipeline-ı birləşdirən orkestrasiya skripti, yeni aşkarlama məntiqi yazmır. **Skor/fusion YOXDUR** (bax bölmə 5, "Mərhələ 5/6 sadələşdirmə qərarı") — hər label birbaşa xam siqnaldan gəlir.
- `run_pipeline(video_path, ..., show_live=False, max_frames=None) -> str`. Hər frame: `tracker.PersonTracker.update_frame()` → `pose_utils.get_keypoints_for_all()` → hər `track_id` üçün `head_pose_yolo.estimate_head_pose_yolo()` (`pose_by_id` dict-ində saxlanılır, HAND RISK cross-check-i üçün hər iki tərəfin istiqaməti lazımdır) + `hand_proximity.HandProximityTracker.update()` + `detector.detect_phones()`/`associate_phones()` → hər `track_id` üçün `_build_labels()` (yerli funksiya, skorsuz, şərt-əsaslı) → `tracker.draw_tracks()` + `visualizer.draw_alert_labels()`. Nəticə `output_dir/<video_adı>_annotated.mp4`-ə yazılır.
- ✅ **`supervision` ilə sadələşdirildi (bax bölmə 9):** video oxuma/yazma `cv2.VideoCapture`/`cv2.VideoWriter` + manual `fps/width/height` + `try/finally` əvəzinə `sv.VideoInfo.from_video_path()` + `sv.get_video_frames_generator(video_path, end=max_frames)` + `with sv.VideoSink(...) as sink: sink.write_frame(frame)`. `show_live=True` üçün `cv2.imshow`/`cv2.waitKey` olduğu kimi qalır (supervision-da ekvivalenti yoxdur).
- `_build_labels(track_id, direction, phones, hand_results, pose_by_id)`: `PhoneBox.owner_id == track_id` → `"PHONE"`; `direction in (SIDE, AWAY)` → `"LOOKING AROUND"`; `track_id` iştirak edən cütdə `_should_flag_hand_risk()` `True` qaytarırsa → `"HAND RISK"`. Heç bir toplama/çəki YOXDUR, hər şərt müstəqil yoxlanılır.
- ✅ **`_should_flag_hand_risk(pair_result, direction_1, direction_2)` — DAİMİ qərar (gaze cross-check):** HAND RISK yalnız `is_sustained=True` VƏ cütün heç olmasa bir tərəfinin istiqaməti SIDE/AWAY olanda (FORWARD-da statik yaxınlıq deyil) tetiklənir. Sınaqdan keçirilib (2 real video, `DSC_0095.MOV`/`DSC_0093(change).MOV`) və saxlanılır — müvəqqəti deyil.
- `HAND_DISTANCE_RATIO=1.5`, `HAND_CONSECUTIVE_FRAMES=10` (fayl başında sabit, `HandProximityConfig`-ə ötürülür) — orijinal dəyərlərdə qalır, gələcəkdə labeled klip-lərlə kalibrasiya oluna bilər.
- ⚠️ **Tarixçə (silinib):** `show_debug` parametri və ona bağlı overlay (hər cütün ortasında `d=`/`c=` mətni, sonra `distance<3.0` filtri) müvəqqəti kalibrasiya məqsədi ilə əlavə olunmuşdu, gaze cross-check-i 2 real videoda sınadıqdan sonra tamamilə silindi (`core/hand_proximity.py`-dəki YALNIZ bu məqsədə xidmət edən `get_consecutive_count()` da silindi, əsas tracking məntiqinə toxunulmadı).
- `show_live: bool = False` (defolt) — yalnız fayla yazır, GUI-siz mühitdə də işləyir. `True` olsa əlavə `cv2.imshow` göstərir.
- `max_frames: int | None = None` — verilsə, yalnız ilk N frame işlənir (sürətli smoke-test üçün, headless mühitdə tam video çox uzun çəkdiyi üçün əlavə olundu).
- **Test:** `ai_vision` mühiti ilə, `Data/Group/v5/DSC_0079(phone).MOV` üzərində, `show_live=False, max_frames=60`. Pipeline uçdan-uca işlədi, `output/DSC_0079(phone)_annotated.mp4` (60 frame, ~9.9MB) yaradıldı.

## 4. Qərar: hand_proximity.py → risk_scorer.py körpüsü (⚠️ LƏĞV EDİLDİ)

`hand_proximity.py` funksional olaraq artıq işləkdir və dəyişdirilməyəcək. Onun üzərinə **nazik bir wrapper funksiyası** əlavə olunacaq ki, ardıcıl `is_sustained=True` frame-ləri diskret `(track_id, qonşu_id, məsafə, müddət, timestamp)` event-ə çevirsin — `risk_scorer.py` bu formatı gözləyəcək. `HandProximityTracker`-in daxili state-inə toxunulmur.

**Tarixi qeyd:** bu körpü icra olunmuşdu (`HandProximityEvent`/`HandProximityEventTracker`), amma Mərhələ 5/6 sadələşdirmə qərarı ilə (bax bölmə 5) `risk_scorer.py` özü silindiyi üçün körpü də lazımsızlaşdı və silindi. Bölmə tarixi kontekst üçün saxlanılır.

## 5. Növbəti addımlar (sıra ilə)

### Mərhələ 5 — Risk Scoring Engine ✅ tamamlandı, sonra ⚠️ LƏĞV EDİLDİ (bax aşağıda "Mərhələ 5/6 sadələşdirmə")

1. ✅ **`analysis/config.py`** — tamamlanıb. İlkin 5 sabit: `YAW_THRESHOLD_DEG=40`, `SHORT_DURATION_S=3`, `LONG_DURATION_S=8`, `YAW_HAND_MULTIPLIER=3`, `YAW_SHOULDER_MULTIPLIER=2` (`HAND_PROXIMITY_PX` istifadəsiz olduğu üçün sonradan silindi, bax bölmə 3). Test edilib. **Placeholder-dir, Mərhələ 7-də labeled clip-lərlə kalibrasiya olunacaq.**

2. ✅ **Event wrapper-lər** — hər iki siqnal mənbəyi üçün tamamlanıb, risk_scorer indi ikisini də diskret event kimi istehlak edə bilər:
   - `hand_proximity.py` daxilində `HandProximityEvent`/`HandProximityEventTracker` (bax bölmə 3). Sintetik self-check (`_demo_event_tracker`) ilə təsdiqlənib.
   - `head_pose_yolo.py` daxilində `YawEvent`/`YawEventTracker` (bax bölmə 3), `YAW_THRESHOLD_DEG`-i `analysis/config.py`-dən istifadə edir. Sintetik self-check (`_demo_yaw_event_tracker`) ilə təsdiqlənib.

3. ✅ **`analysis/risk_scorer.py`** — tamamlanıb (bax bölmə 3 ətraflı təsvir üçün). Qeydlər:
   - **Tip-A əhatəsi (v1):** yalnız telefon, planlaşdırıldığı kimi.
   - **Shoulder tilt** modulu hələ yoxdur — `SHOULDER_TILT_POINTS` və `YAW_SHOULDER_MULTIPLIER` config-də var, amma `risk_scorer.py`-də `TODO` şərhi ilə istifadəsiz saxlanılıb.
   - **Multiplier:** yalnız Yaw+Hand (`×YAW_HAND_MULTIPLIER`) implementasiya olunub, boolean şərtlə (kompaund olunmur). Yaw+Shoulder shoulder modulu yazılanda əlavə olunacaq. Qonşu-korrelyasiya multiplier-i əvvəlki kimi bu mərhələdə yazılmır (seat mapping yoxdur, future work).
   - **Kateqoriya/tavan/Gate-0** spesifikasiyaya tam uyğun implementasiya olunub, sintetik self-check ilə təsdiqlənib.

### Mərhələ 6 — Vizuallaşdırma və Hesabat ✅ tamamlandı, sonra qismən ⚠️ LƏĞV EDİLDİ (draw_risk_label/report_generator.py silindi, draw_hand_proximity qaldı — bax aşağıda)

4. ✅ **`reporting/visualizer.py`** — `draw_risk_label()` əlavə olundu (bax bölmə 3). Rəng kodu, primary_evidence göstərilməsi, CRITICAL blink effekti — hamısı spesifikasiyaya uyğun. Skor rəqəmi yazılmır. Sintetik self-check (`_demo_visualizer`) ilə təsdiqlənib.

5. ✅ **`reporting/report_generator.py`** — tamamlanıb (bax bölmə 3). "Sübut piramidası" (əsas/ikinci dərəcəli, timestamp-ə görə sıralı) həm dict (JSON-serializable), həm mətn formatında. Sintetik self-check (`_demo_report_generator`) ilə təsdiqlənib.

### Mərhələ 5/6 — Sadələşdirmə qərarı (silinmə)

**Qərar:** Mərhələ 5 (Risk Scoring Engine) və Mərhələ 6-nın (Vizuallaşdırma/Hesabat) bir hissəsi geri götürülür — kaskadlı skor/hesabat sistemi çıxarılır, layihə sadə "hər frame-də xam siqnal-əsaslı label göstərmə" formasına qaytarılır (bax aşağıda yeni `main.py` spesifikasiyası).

Aşağıdakılar işçi qovluqdan silinir. **GitHub tarixçəsində qalır** — lazım olsa `git log -- <fayl>` ilə commit tapılıb `git checkout <commit> -- <fayl>` ilə bərpa oluna bilər:

- `analysis/risk_scorer.py` — bütün fayl
- `reporting/report_generator.py` — bütün fayl
- `analysis/config.py` — bütün fayl. Yeganə istehlakçıları (`risk_scorer.py`, `head_pose_yolo.py`-dəki `YawEventTracker`) də silinir, ona görə `analysis/` qovluğu boş qalarsa qovluq özü də silinir.
- `core/head_pose_yolo.py`-dan: `YawEvent` dataclass, `YawEventTracker` class, `_demo_yaw_event_tracker`, `config.py`-ə aid `sys.path`/import sətirləri, `__main__`-dəki demo çağırışı.
- `core/hand_proximity.py`-dan: `HandProximityEvent` dataclass, `HandProximityEventTracker` class, `_demo_event_tracker`, `__main__` bloku (tamamilə boşaldığı üçün).
- `reporting/visualizer.py`-dan: `draw_risk_label`, onun `analysis.risk_scorer`-dən `RiskCategory`/`RiskResult` importu, yalnız bu import üçün əlavə edilmiş `analysis/` `sys.path` sətri, `_demo_visualizer`, `__main__` bloku (tamamilə boşaldığı üçün).

**Toxunulmayanlar** (sadəcə xam siqnal mənbəyi kimi qalır): `HeadPoseResult`, `GazeConfig`, `estimate_yaw_from_keypoints`, `classify_direction`, `estimate_head_pose_yolo`, `draw_head_pose_yolo` (`head_pose_yolo.py`); `run_yolo_pose_test` (məzmunu dəyişmədi, sonradan `tests/test_head_pose_yolo.py`-ə köçürüldü — bax yuxarı); `HandProximityConfig`, `PairProximityResult`, `HandProximityTracker` (`hand_proximity.py`); `draw_hand_proximity` (`visualizer.py`); `tests/test_hand_proximity.py` (dəyişmir).

### Mərhələ 7 — Validation (SADƏLƏŞDİRİLDİ)

**Qərar:** avtomatik `validation/validator.py` (label parsing, precision/recall) YAZILMAYACAQ. Bunun əvəzinə: istifadəçi özü bildiyi videoları pipeline-dan keçirir, nəticəyə (bbox label) baxıb əl ilə səhvləri tapır, qeyd edir və düzəldir. Rəqəmli metrik yerinə keyfiyyət təsviri (aşkarlanan problemlər + düzəlişlər) — bu, layihənin data-leakage qeydi ilə eyni şəffaflıq prinsipidir.

### `main.py` — ✅ tamamlandı (sadə spesifikasiya)

Bax bölmə 3 ətraflı təsvir üçün (imza, `_build_labels()` şərtləri, test nəticəsi). Xülasə:

**Fəlsəfə dəyişikliyi:** skor/fusion YOXDUR. Hər track_id üçün xam siqnallardan birbaşa mətn label siyahısı qurulur, heç bir çəkili toplama və ya kateqoriyalaşdırma edilmir.

Hər frame: `persons = person_tracker.update_frame(frame)` → `keypoints_by_id = pose_utils.get_keypoints_for_all(...)` → hər `track_id` üçün `estimate_head_pose_yolo(...)` ilə `HeadPoseResult` (`direction`, mövcud `GazeConfig` hədləri) → `hand_tracker.update(keypoints_by_id)` → `PairProximityResult` siyahısı (`is_sustained`) → `detect_phones()` + `associate_phones()` → `PhoneBox` siyahısı (`owner_id`) → `_build_labels()` → `draw_tracks()` + `visualizer.draw_alert_labels()`.

Çıxış: annotasiya olunmuş video `cv2.VideoWriter` ilə `output/<video_adı>_annotated.mp4`-ə yazılır. `show_live: bool = False` (defolt) — GUI-siz mühitdə də işləsin deyə. `ai_vision` mühitində, `DSC_0079(phone).MOV` üzərində, 60 frame ilə sınaqdan keçirilib.

## 6. İş prinsipləri (mütləq riayət olunmalı)

- Kiçik, tək-məqsədli funksiyalar, monolit blok yox.
- Hər yeni modul üçün ayrıca, real video üzərində test/demo bölməsi.
- Modullar arası aydın sərhəd: ortaq inference (`pose_utils`) → tapşırığa-spesifik istehlakçılar.
- Video-level train/val split (frame-level YOX) — data leakage-dan qaçmaq üçün.
- Bir modul tam işləyib test olunmadan növbətiyə keçilmir.
- **Mövcud işlək kodu (tracker.py, detector.py, pose_utils.py, head_pose_yolo.py, hand_proximity.py) struktur/ad uyğunluğu naminə YENİDƏN YAZILMIR.** Yeni modullar mövcud interfeyslərə uyğunlaşır, əksinə yox.
- **Core aşkarlama modulları (`core/*.py`) yalnız aşkarlama məntiqini saxlayır.** Vizual debug funksiyaları (`draw_*`) `reporting/visualizer.py`-ə, tam-pipeline test/demo skriptləri (video açan, `cv2.imshow` istifadə edən, hardcoded yol olan) ayrıca `tests/` və ya `scripts/` qovluğuna aiddir. Modeldən/videodan asılı olmayan sürətli self-check funksiyaları (məs. `_demo_event_tracker`) core faylda qala bilər — bunlar yüngüldür və modulun düzgünlüyünü sürətli yoxlayır.

## 7. Mühit (Environment)

- `cv2`/`ultralytics` yalnız conda mühitində mövcuddur: **`C:\Users\user\anaconda3\envs\ai_vision`**. Sistem PATH-dəki `python`/`py` bu paketləri görmür.
- Test/skript işə salarkən mütləq bu mühiti aktivləşdirin, əks halda import xətaları yaranır.

## 8. Açıq problemlər / gələcək iş

- Telefon aşkarlanması hələ də açıq problemdir (kiçik obyekt ölçüsü, ~15–25px) — crop-based fallback hələ yazılmayıb, SAHI (slicing inference) gələcəkdə nəzərdən keçirilə bilər.
- Seat mapping, instructor dashboard, VLM-based izahat, audio-based danışıq aşkarlanması, üz blurlama, edge optimizasiyası — "future work" bölməsinə, hazırkı əhatədən kənarda.

## 9. `supervision` kitabxanası — qəbul edilən/rədd edilən namizədlər

Layihədə artıq quraşdırılmış (`ai_vision` mühitində, `rfdetr`-in asılılığı kimi gəlib) Roboflow-un `supervision` (`pip install supervision`, `import supervision as sv`) kitabxanasından YALNIZ infrastruktur/annotation səviyyəsində, aşağıdakı 3 dəyişiklik qəbul edilib. Domain məntiqi (yaw hesablama, hand-proximity heuristikası, gaze cross-check, telefon-sahib təyini) TOXUNULMAYIB — `supervision` bunları əvəz edə bilməz, ümumi CV kitabxanasıdır.

**✅ Tətbiq olunanlar:**
1. **`main.py` video I/O** — `cv2.VideoCapture`/`cv2.VideoWriter` + manual `fps/width/height` + `try/finally` əvəzinə `sv.VideoInfo.from_video_path()` + `sv.get_video_frames_generator(video_path, end=max_frames)` + `with sv.VideoSink(...) as sink: sink.write_frame(frame)`. `show_live`/`cv2.imshow` toxunulmayıb.
2. **`core/tracker.py:extract_tracks()`** — daxili parsing `sv.Detections.from_ultralytics(results[0])` ilə edilir, çıxış yenə `list[TrackedBox]`-dır. `TrackedBox`-un `.x1/.y1/.x2/.y2/.track_id` interfeysi HEÇ YERDƏ (detector.py, pose_utils.py, hand_proximity.py, main.py) dəyişməyib.
3. **`core/tracker.py:draw_tracks()`** — `_ID_COLOR_CACHE`/`_color_for_id` hand-rolled rəng-keşi silinib, `sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)` ilə əvəzlənib (yalnız çəkmə anında müvəqqəti `sv.Detections` qurulur, `TrackedBox`-un özü dəyişməyib).
4. **`core/detector.py:detect_phones()`** — eyni nümunə: manual `.xyxy`/`.conf` parsing VƏ `if len(results[0].boxes) == 0: return []` yoxlaması `sv.Detections.from_ultralytics(results[0])` ilə əvəzləndi (boş `zip()` avtomatik `[]` verir). `PhoneBox` (owner_id daxil) və `associate_phones()` TAM TOXUNULMAZ.

Hər 4-ü `ai_vision` mühitində real video üzərində (`DSC_0079(phone).MOV`) sınaqdan keçirilib, mövcud self-check (`tests/test_visualizer.py`) və bütün import-lar təsdiqlənib.

**❌ Bilərəkdən tətbiq edilməyənlər (əvvəlki analiz hesabatlarından, rədd edilib — gələcəkdə təkrar araşdırılmasın):**
- **`sv.ByteTrack`** — `v0.28.0`-dan **deprecated**, `v0.30.0`-da siləcək. Üstəlik, `PersonTracker.update_frame()` artıq ultralytics-in öz daxili `.track(tracker="bytetrack.yaml")` çağırışından istifadə edir — `sv.ByteTrack` əlavə etmək ikiqat tracking (yaxud silinməkdə olan API-yə keçid) olardı.
- **`reporting/visualizer.py` — `draw_alert_labels`/`draw_hand_proximity`** — `sv.LabelAnnotator`-ın `smart_position` funksiyası FƏRQLİ detection-ların label-lərinin bir-birinə DƏYMƏMƏSİ üçündür, bizim problemimiz olan "bir adamın 2-3 label-i öz box-u daxilində şaquli sıralansın" məsələsini həll etmir (`LabelAnnotator` hər detection üçün YALNIZ BİR label string qəbul edir). `draw_hand_proximity`-nin iki fərqli detection arasında xətt çəkməsinin isə supervision-da heç bir ekvivalenti yoxdur. Hər ikisi olduğu kimi saxlanılıb.
- **`core/pose_utils.py:get_pose_keypoints()` → `sv.KeyPoints.from_ultralytics`** — kitabxananın mənbə kodu yoxlanılıb: `sv.KeyPoints` MÖVCUDDUR, amma `.xy` (n,17,2) və `.keypoint_confidence` (n,17) AYRI-AYRI massivlərdə qaytarır, heç vaxt bizim tələb etdiyimiz birləşmiş `(17,3)` `[x,y,conf]` formatına (`results[0].keypoints.data[0]`, `head_pose_yolo.py`/`hand_proximity.py`-nın hər yerdə `keypoints[NOSE]` kimi indekslədiyi format) gətirilmir. Adopt etmək mövcud 1 sətri sadələşdirmək əvəzinə əlavə `np.concatenate`-lə yenidən-birləşdirmə tələb edərdi — **format uyğunsuzluğu səbəbindən rədd edilib**, sadələşdirmə deyil.

Gələcəkdə "niyə supervision hər yerdə istifadə olunmayıb" sualı yaranarsa — cavab yuxarıdakı 3 bənddədir.