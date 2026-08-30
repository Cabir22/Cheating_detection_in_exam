# Layihə İzahı

## Ümumi baxış

Bu layihə imtahan zalının küncdən quraşdırılmış, geniş bucaqlı kamera görüntüsünü analiz edərək tələbələr arasında köçürmə/şübhəli hərəkət hallarını (telefon istifadəsi, ətrafa baxma, əl vasitəsilə kağız/qeyd ötürülməsi) real-vaxta yaxın aşkarlayan bir kompüter-görmə sistemidir. Sistem eyni anda 10+ tələbəni izləyə bilir, hər şəxsə fərdi ID verir və onun davranışını frame-bə-frame qiymətləndirir.

Layihə hazırda "sadə, skorsuz" fəlsəfə ilə işləyir: hər aşkarlama modulu (telefon, baxış istiqaməti, əl yaxınlığı) öz xam siqnalını verir, bu siqnallar heç bir çəkili toplama və ya risk balı hesablanmadan birbaşa şərti (if-based) qaydalarla mətn etiketlərinə (məs. "PHONE", "LOOKING AROUND", "HAND RISK") çevrilir və video üzərində bounding box-ların içində göstərilir. Əvvəllər planlaşdırılmış daha mürəkkəb "risk skorlama mühərriki" tam işlənib sonra bilərəkdən geri götürülüb (bax "Natamam/planlaşdırılan hissələr").

## core/ qovluğu

### tracker.py
- **Məqsəd:** Kadr üzərində insanları (tələbələri) aşkarlamaq və onları frame-lər arası ardıcıl ID-lərlə izləmək (tracking). Bütün digər modulların əsasını təşkil edir.
- **Əsas funksiyalar/class-lar:**
  - `TrackedBox` (dataclass) — bir şəxsin bir frame-dəki vəziyyəti: `track_id, x1, y1, x2, y2, conf`. Demək olar bütün digər fayllarda istifadə olunan ortaq məlumat strukturudur.
  - `PersonTracker` (class) — YOLO modelini (`yolo26m.pt`) yükləyir, ByteTrack ilə birlikdə `update_frame(frame)` metodu vasitəsilə hər kadr üçün `list[TrackedBox]` qaytarır (yalnız "insan" sinfi, `classes=[0]`).
  - `extract_tracks(results)` — YOLO-nun xam nəticəsini (`supervision` kitabxanası ilə) təmiz `TrackedBox` siyahısına çevirir.
  - `draw_tracks(frame, tracks)` — hər tracked şəxsin ətrafına bbox və ID yazısı çəkir (vizual debug).
- **Əlaqəli fayllar:** Heç bir digər core modulundan asılı deyil (ən aşağı səviyyəli/kök modul). `TrackedBox`-ı `detector.py`, `pose_utils.py`, `main.py` import edir.

### detector.py
- **Məqsəd:** Kadrda telefon aşkarlamaq və tapılan telefonu ən yaxın tələbəyə (sahibinə) bağlamaq.
- **Əsas funksiyalar/class-lar:**
  - `PhoneBox` (dataclass) — `x1, y1, x2, y2, conf, owner_id` (owner_id sonradan təyin olunur).
  - `detect_phones(model, frame, conf)` — ayrıca telefon-aşkarlama YOLO modeli ilə kadrda telefonları tapır, `list[PhoneBox]` qaytarır.
  - `associate_phones(phones, persons, max_distance)` — hər telefonu, mərkəzi ən yaxın olan `TrackedBox`-un `track_id`-sinə bağlayır (məsafə `max_distance`-i keçərsə sahib təyin olunmur).
- **Əlaqəli fayllar:** `tracker.py`-dan `TrackedBox` import edir. `pose_utils`/`head_pose_yolo` ilə birbaşa əlaqəsi yoxdur — öz ayrıca YOLO modelini (fərqli çəkilər, `best11m.pt`) işlədən müstəqil pipeline-dır. Nəticələri `main.py`-da tracker-in çıxışı ilə birləşdirilir.

### pose_utils.py
- **Məqsəd:** YOLO-Pose modelini bir dəfə çağırıb, çıxan skelet nöqtələrini (keypoint) digər iki analiz moduluna (baş istiqaməti və əl yaxınlığı) paylaşan ortaq təməl. Dublikat model çağırışının qarşısını alır.
- **Əsas funksiyalar/class-lar:**
  - Keypoint indeksləri (`NOSE=0`, `LEFT_EYE=1`, … `RIGHT_WRIST=10`) — COCO formatı.
  - `PoseConfig` (dataclass) — keypoint görünürlük həddi, pose model confidence-i, crop padding.
  - `is_visible(keypoint, cfg)` — keypoint-in confidence-i həddi keçirmi yoxlayır.
  - `crop_person_for_pose(frame, person, cfg)` — şəxsin bbox-unu (bir az padding ilə) kadrdan kəsir (pose modelinə bütün bədən lazımdır, yalnız üz yox).
  - `get_pose_keypoints(pose_model, person_crop, cfg)` — kəsilmiş şəkil üzərində YOLO-Pose işlədir, `(17,3)` numpy array (`[x,y,conf]`) qaytarır.
  - `get_keypoints_for_person(...)` — crop + predict + koordinatları geri tam kadr koordinatına çevirmə, tək giriş nöqtəsi.
  - `get_keypoints_for_all(pose_model, frame, persons, cfg)` — bütün tracked şəxslər üçün yuxarıdakını tətbiq edib `dict[track_id, keypoints]` qaytarır.
- **Əlaqəli fayllar:** `tracker.py`-dan `TrackedBox` import edir. `head_pose_yolo.py` və `hand_proximity.py` hər ikisi bu modulun çıxışını (keypoint-ləri) giriş kimi istifadə edir.

### head_pose_yolo.py
- **Məqsəd:** Keypoint-lərdən (burun, çiyin/göz) baş çevrilmə bucağını (yaw) hesablayıb, şəxsin "irəli baxır / yana baxır / kənara baxır" kateqoriyasını təyin etmək — köçürmə zamanı ətrafa baxma siqnalı.
- **Əsas funksiyalar/class-lar:**
  - `GazeDirection` (Enum: `FORWARD, SIDE, AWAY, FACE_NOT_VISIBLE`).
  - `HeadPoseResult` (dataclass) — `track_id, yaw, direction`.
  - `GazeConfig` — `side_threshold=25°`, `away_threshold=55°`.
  - `estimate_yaw_from_keypoints(keypoints, pose_cfg)` — burun ilə çiyin (və ya göz, çiyin görünməzsə) mərkəzi arasındakı yerdəyişməyə əsasən yaw bucağını hesablayır.
  - `classify_direction(yaw, gaze_cfg)` — bucağı `GazeDirection`-a çevirir.
  - `estimate_head_pose_yolo(...)` — hər şeyi birləşdirən əsas funksiya, `keypoints` artıq hesablanıbsa (pose_utils-dan) təkrar model çağırmır.
  - `draw_head_pose_yolo(...)` — vizual debug (bucaq + istiqamət yazısı).
- **Əlaqəli fayllar:** `pose_utils.py`-dan keypoint alır (birbaşa və ya `main.py` üzərindən paylaşılmış), `tracker.py`-dan `TrackedBox`. `main.py`-da hər `track_id` üçün çağırılır və nəticə həm "LOOKING AROUND" etiketi, həm də əl-yaxınlığı siqnalının "gaze cross-check"-i üçün istifadə olunur.

### hand_proximity.py
- **Məqsəd:** İki tələbənin biləklərinin (wrist) bir-birinə yaxınlaşmasını izləməklə kağız/qeyd ötürülməsi kimi hərəkətlərin siqnalını vermək.
- **Əsas funksiyalar/class-lar:**
  - `HandProximityConfig` — `distance_threshold_ratio=1.5` (çiyin eninə nisbətən yaxınlıq həddi), `required_consecutive_frames=10`.
  - `PairProximityResult` (dataclass) — `track_id_1, track_id_2, distance, is_sustained`.
  - `_shoulder_width`, `_visible_wrists`, `_closest_hand_pair` — köməkçi funksiyalar (normallaşdırma üçün çiyin eni, görünən biləklər, ən yaxın əl cütü).
  - `HandProximityTracker` (class) — frame-lər arası state saxlayır (`_consecutive_counts` dict): hər tələbə cütü üçün neçə ardıcıl frame-dir yaxın qaldığını sayır, `required_consecutive_frames`-i keçəndə `is_sustained=True` verir (təsadüfi tək-frame sıçrayışların siqnal verməsinin qarşısını alır). `update(keypoints_by_id)` metodu hər frame-də bütün cütləri yoxlayıb `list[PairProximityResult]` qaytarır.
- **Əlaqəli fayllar:** `pose_utils.py`-nın keypoint çıxışını istifadə edir, öz pose model çağırışı yoxdur. Vizual çəkmə funksiyası (`draw_hand_proximity`) `reporting/visualizer.py`-ə köçürülüb. `main.py`-da nəticə "HAND RISK" etiketinin (baş istiqaməti ilə birgə) əsasını təşkil edir.

## main.py

Proqram işə düşəndə (`run_pipeline(video_path, ...)` çağırılanda) baş verənlər ardıcıllıqla:

1. **Modulların idxalı:** `core/` və `reporting/` qovluqları `sys.path`-ə əlavə olunur (layihədə rəsmi paket strukturu yoxdur), sonra `pose_utils`, `tracker.PersonTracker/draw_tracks`, `head_pose_yolo.estimate_head_pose_yolo/GazeDirection`, `hand_proximity.HandProximityTracker/HandProximityConfig`, `detector.detect_phones/associate_phones`, `visualizer.draw_alert_labels` import olunur.

2. **Başlanğıc qurulum (`run_pipeline` çağırılanda bir dəfə):**
   - `PersonTracker` (insan modeli, `yolo26m.pt`, defolt yol) yaradılır.
   - Pose modeli (`yolov8m-pose.pt`) və telefon modeli (`best11m.pt`) ayrıca `YOLO(...)` obyektləri kimi GPU-ya (`"cuda"`) yüklənir.
   - `HandProximityTracker` sabitlərlə (`HAND_DISTANCE_RATIO=1.5`, `HAND_CONSECUTIVE_FRAMES=10`, fayl başında təyin olunub) yaradılır.
   - `sv.VideoInfo.from_video_path()` ilə video meta-data (fps/ölçü) oxunur, çıxış qovluğu (`output_dir`, defolt `"output"`) yaradılır.

3. **Hər frame üçün (`sv.get_video_frames_generator` ilə oxunur, `sv.VideoSink` ilə yazılır):**
   - `person_tracker.update_frame(frame)` → `list[TrackedBox]` (tələbələr).
   - `pose_utils.get_keypoints_for_all(...)` → hər tələbə üçün keypoint (bir dəfə hesablanır, aşağıdakı iki modul paylaşır).
   - `hand_tracker.update(keypoints_by_id)` → `list[PairProximityResult]` (əl-yaxınlığı siqnalları).
   - `detect_phones(...)` + `associate_phones(...)` → `list[PhoneBox]` (owner_id ilə).
   - Hər tələbə üçün `estimate_head_pose_yolo(...)` → `HeadPoseResult` (baş istiqaməti), `pose_by_id` dict-ində saxlanılır.
   - `draw_tracks(frame, persons)` — bbox + ID çəkilir.
   - Hər tələbə üçün `_build_labels(track_id, direction, phones, hand_results, pose_by_id)` — yerli funksiya, şərt-əsaslı (skorsuz) qaydalarla etiket siyahısı qurur:
     - Telefonun `owner_id`-si bu `track_id`-yə uyğundursa → `"PHONE"`.
     - Baş istiqaməti `SIDE`/`AWAY`-dırsa → `"LOOKING AROUND"`.
     - Bu tələbənin iştirak etdiyi hər hansı əl-cütü `is_sustained=True` OLUB VƏ cütün heç olmasa bir tərəfinin baxışı `SIDE`/`AWAY`-dırsa (statik, hər ikisi irəli baxaraq yazan cütlər tetiklənmir — `_should_flag_hand_risk` funksiyası bunu yoxlayır) → `"HAND RISK"`.
   - Etiketlər varsa `visualizer.draw_alert_labels(frame, person, labels)` ilə bbox daxilində göstərilir.
   - Frame `sink.write_frame(frame)` ilə çıxış videosuna yazılır (`show_live=True` olsa əlavə `cv2.imshow` da göstərir).

4. **Giriş/çıxış və parametrlər:**
   - **Giriş:** video faylının yolu (`video_path`), üç model yolu (insan/pose/telefon, defolt olaraq `models/` qovluğunda), `tracker_cfg="bytetrack.yaml"`, `output_dir="output"`, `show_live=False`, `max_frames=None` (verilsə yalnız ilk N frame işlənir — sürətli sınaq üçün).
   - **Çıxış:** annotasiya olunmuş video, `output_dir/<video_adı>_annotated.mp4` yolunda; funksiya bu yolu string olaraq qaytarır.
   - Skript birbaşa işə salınanda (`__main__`) əmr sətri arqumenti kimi video yolu qəbul edir; alternativ olaraq `run.py` müxtəlif test video yollarını əl ilə seçib `main.run_pipeline(...)` çağırır.

## Digər qovluqlar/fayllar

- **`reporting/visualizer.py`** — core modulların aşkarlama nəticələrini kadr üzərinə çəkən vizual funksiyalar: `draw_hand_proximity()` (iki şəxs arasında sustained əl-yaxınlığı xətti), `draw_alert_labels()` (hər tələbənin bbox-u daxilində "PHONE"/"LOOKING AROUND"/"HAND RISK" kimi etiketləri rəngli fon zolağı ilə göstərir — telefon qırmızı, digərləri sarı). Aşkarlama məntiqi burada yoxdur, yalnız göstərmə.
- **`tests/`** — hər core modul üçün ayrıca, real video/model üzərində işləyən test/demo skriptləri (`test_tracker.py`, `test_detector.py`, `test_head_pose_yolo.py`, `test_hand_proximity.py`) və modeldən asılı olmayan sintetik self-check (`test_visualizer.py`, uydurma bbox-larla `draw_alert_labels`-i yoxlayır). Avtomatlaşdırılmış test paketi deyil — çoxu `cv2.imshow` ilə əl ilə vizual yoxlama üçündür.
- **`models/`** — layihədə istifadə olunan bütün YOLO çəki fayllarını (`.pt`) saxlayır: insan aşkarlama (`yolo26m.pt`), pose (`yolov8m-pose.pt`), telefon aşkarlama (`best11m.pt`, üstəlik sınaq/alternativ çəkilər: `best_kaggle.pt`, `bestv8m.pt`, `rtdetr-l.pt`, `yolo11m.pt`, `yolo11x.pt`, `yolov8m.pt`, `yolov8n.pt`).
- **`Data/`** — xam/etiketli video və şəkil materialları: `Group/` (küncdən çəkilmiş qrup videoları, alt-qovluqlar `v2/v4/v5` kimi partiyalara bölünüb, məs. `DSC_0079(phone).MOV`), `Solo/` (yaxın-plan tək-şəxs klipləri, `images_train/`, `v1/`, `v2/` alt-qovluqları ilə), `Picture/` (ayrıca statik şəkillər, məs. `DSC_0081_img001.png`). CLAUDE.md-də qeyd olunduğu kimi, etiketli klip data ground-truth yoxlama üçündür, model təliminə DAXİL EDİLMİR.
- **`run.py`** (kök qovluqda) — `main.run_pipeline()`-i əl ilə müxtəlif test videoları üzərində (şərh olunmuş siyahıdan biri seçilərək) `show_live=True` ilə işə salan kiçik köməkçi skript, `main.py`-ın CLI interfeysinə alternativ sürətli sınaq yoludur.
- **`output/`** — pipeline-ın yazdığı annotasiya olunmuş video/şəkil nəticələri (məs. `test_visualizer_labels.png`, `*_annotated.mp4`).
- CLAUDE.md-ə görə əvvəllər mövcud olmuş `analysis/` (risk_scorer.py, config.py) və `reporting/report_generator.py` bilərəkdən silinib (bax aşağıda "Natamam/planlaşdırılan hissələr"), hazırkı qovluq strukturunda yoxdur.

## Data axını

```
Video fayl (Data/.../*.MOV)
        │
        ▼
sv.get_video_frames_generator  →  hər frame (np.ndarray)
        │
        ├──► PersonTracker.update_frame(frame) ──► list[TrackedBox] (tələbələr + ID-lər)
        │              │
        │              ▼
        ├──► pose_utils.get_keypoints_for_all(pose_model, frame, persons)
        │              │  → dict[track_id, keypoints(17,3)]
        │              │
        │        ┌─────┴─────┐
        │        ▼           ▼
        │  head_pose_yolo   hand_proximity.HandProximityTracker.update(...)
        │  .estimate_head_    │  → list[PairProximityResult] (is_sustained)
        │  pose_yolo(...)     │
        │  → dict[track_id,   │
        │     HeadPoseResult] │
        │        │            │
        ├──► detector.detect_phones(frame) + associate_phones(persons)
        │              │  → list[PhoneBox] (owner_id təyin olunmuş)
        │              ▼
        └──► main._build_labels(track_id, direction, phones, hand_results, pose_by_id)
                       │  → list[str] (məs. ["PHONE", "HAND RISK"])
                       ▼
        tracker.draw_tracks(frame) + visualizer.draw_alert_labels(frame, person, labels)
                       │
                       ▼
        sv.VideoSink.write_frame(frame)
                       │
                       ▼
        output/<video_adı>_annotated.mp4
```

Qısaca: video → (paralel olaraq) insan izləmə + pose keypoint çıxarma → keypoint-lər baş-istiqaməti VƏ əl-yaxınlığı analizinə paylaşılır → ayrıca telefon aşkarlama pipeline-ı işə düşür → bütün siqnallar hər tələbə üçün şərti qaydalarla mətn etiketlərinə çevrilir → kadr üzərinə çəkilir → annotasiya olunmuş video fayla yazılır. Skor/agregasiya mərhələsi yoxdur — hər etiket öz xam siqnalından birbaşa gəlir.

## Natamam/planlaşdırılan hissələr

- **Telefon aşkarlanması hələ açıq problemdir** — kiçik obyekt ölçüsü (~15–25px) səbəbindən dəqiqlik məhduddur. Planlaşdırılan crop-based fallback (əvvəl şəxsi kəsib sonra o crop daxilində telefon axtarmaq) hələ yazılmayıb; SAHI (slicing inference) gələcəkdə nəzərdən keçirilə bilər.
- **Risk Scoring Engine (skorlama mühərriki)** — əvvəllər tam yazılmış (`analysis/risk_scorer.py`, `analysis/config.py`, `HandProximityEvent`/`YawEvent` "event" körpüləri, `reporting/report_generator.py`, `draw_risk_label()`), sonra layihə "sadə, xam-siqnal-əsaslı label" fəlsəfəsinə qayıtmaq qərarı ilə tamamilə geri götürülüb və kod bazasından silinib (GitHub tarixçəsində qalır). Səbəb sənədləşdirilmiş sadələşdirmə qərarıdır, texniki problem deyil.
- **Shoulder tilt (çiyin əyilməsi) modulu** — planlaşdırılmış, lakin heç vaxt yazılmayıb (silinmiş `config.py`-də ona aid sabitlər var idi, istifadəçisi olmadan).
- **Avtomatlaşdırılmış validation (`validation/validator.py`, precision/recall metrikləri)** — bilərəkdən YAZILMAYACAQ qərarı verilib; əvəzinə istifadəçi videoları əl ilə pipeline-dan keçirib nəticəyə baxaraq keyfiyyəti qiymətləndirir.
- **Seat mapping, instructor dashboard, VLM-əsaslı izahat, audio-əsaslı danışıq aşkarlanması, üz blurlama, edge optimizasiyası** — layihənin hazırkı əhatəsindən kənarda, gələcək iş kimi qeyd olunub.
