# Instructor Notes — Lab 08: Text Mining and Sentiment Classification

## 1. ข้อมูลแลป

| รายการ | รายละเอียด |
|---|---|
| รายวิชา | Advanced Machine Learning |
| บทเรียน | Data Mining & Web/Text Processing |
| ชื่อแลป | Customer Feedback Sentiment Classification with TF-IDF |
| ระยะเวลา | ไม่เกิน 90 นาที |
| รูปแบบ | Google Colab + Guided Notebook |
| ระดับผู้เรียน | นักศึกษาที่มีพื้นฐาน Python, pandas และ classification เบื้องต้น |
| ภาษา Dataset | ภาษาไทย |
| Task | Binary Text Classification |
| Labels | `positive`, `negative` |
| Model หลัก | TF-IDF + Logistic Regression |
| Evaluation | Accuracy, Precision, Recall, F1-score, Confusion Matrix |
| Dataset สำหรับนักศึกษา | `customer_feedback_sentiment.csv` |
| Dataset Master | `customer_feedback_sentiment_full.csv` |
| Notebook นักศึกษา | `01_text_mining_lab_student.ipynb` |
| Notebook เฉลย | `01_text_mining_lab_solution.ipynb` |

---

## 2. วัตถุประสงค์การเรียนรู้

เมื่อจบแลป นักศึกษาควรสามารถ:

1. โหลดและสำรวจข้อมูลข้อความในรูปแบบ CSV ได้
2. ตรวจ missing values, duplicates และ class distribution ได้
3. ทำ text cleaning ขั้นพื้นฐานได้
4. อธิบายความต่างของ raw text และ clean text ได้
5. แบ่งข้อมูลเป็น train/test set ด้วย `stratify=y` ได้
6. ใช้ TF-IDF เพื่อแปลงข้อความเป็น features ได้
7. ฝึก Logistic Regression สำหรับ sentiment classification ได้
8. อ่าน Accuracy, Precision, Recall และ F1-score ได้
9. อ่านและตีความ Confusion Matrix ได้
10. วิเคราะห์ตัวอย่างที่โมเดลทำนายผิดได้
11. เชื่อมผลโมเดลกับกรณีใช้งาน customer support ได้

---

## 3. สถานการณ์ของแลป

ให้นักศึกษารับบทเป็น Data Analyst ของบริษัทที่ต้องการคัดกรอง
customer feedback จากข้อความภาษาไทย

เป้าหมายคือสร้าง prototype ที่ช่วยจำแนกข้อความเป็น:

- `positive` — ความคิดเห็นเชิงบวก
- `negative` — ความคิดเห็นเชิงลบ

ผลลัพธ์ของโมเดลสามารถใช้เพื่อ:

- ส่งต่อ feedback เชิงลบให้ทีม customer support
- จัดลำดับข้อความที่ควรตอบก่อน
- วิเคราะห์คุณภาพบริการในภาพรวม
- ใช้เป็น baseline ก่อนพัฒนาโมเดล NLP ที่ซับซ้อนขึ้น

> หมายเหตุ:
> Dataset นี้เป็นข้อความจาก social media ที่นำมาใช้แทน customer feedback
> เพื่อการเรียนรู้ ไม่ใช่ข้อมูลรีวิวสินค้า marketplace โดยตรง

---

## 4. ไฟล์และโครงสร้างโฟลเดอร์

```text
ML_Lab_08_Text_Mining/
│
├── student/
│   ├── 01_text_mining_lab_student.ipynb
│   ├── customer_feedback_sentiment.csv
│   ├── README_student.md
│   └── requirements.txt
│
└── instructor/
    ├── 01_text_mining_lab_solution.ipynb
    ├── customer_feedback_sentiment_full.csv
    ├── customer_feedback_sentiment_backup.csv
    ├── prepare_lab_dataset.ipynb
    ├── instructor_notes.md
    └── lab_metadata.json
```

### สิ่งที่แชร์ให้นักศึกษา

แชร์เฉพาะโฟลเดอร์:

```text
student/
```

สิทธิ์ที่แนะนำ:

```text
Anyone with the link → Viewer
```

ห้ามแชร์โฟลเดอร์ `instructor/` ก่อนนักศึกษาส่งงาน

---

## 5. Dataset Information

### แหล่งข้อมูล

- Dataset: Wisesight Sentiment Corpus
- Source: PyThaiNLP / Wisesight
- ภาษา: ไทย
- ลักษณะข้อความ: social media text
- Labels ต้นทาง: `pos`, `neg`, `neu`, `q`
- Labels ที่ใช้ในแลป: `positive`, `negative`
- License ของ annotations/dataset: CC0-1.0

### การ map labels

| Original Label | Label ในแลป |
|---|---|
| `pos` | `positive` |
| `neg` | `negative` |
| `neu` | ไม่ใช้ |
| `q` | ไม่ใช้ |

### Dataset สำหรับนักศึกษา

ไฟล์:

```text
customer_feedback_sentiment.csv
```

โครงสร้าง:

| Column | คำอธิบาย |
|---|---|
| `review_id` | รหัสข้อความ |
| `review_text` | ข้อความดิบก่อน preprocessing |
| `sentiment` | `positive` หรือ `negative` |
| `source` | แหล่งข้อมูล |

ขนาดที่แนะนำ:

| Sentiment | จำนวน |
|---|---:|
| Positive | 400 |
| Negative | 400 |
| รวม | 800 |

### Dataset Master สำหรับผู้สอน

ไฟล์:

```text
customer_feedback_sentiment_full.csv
```

ใช้สำหรับ:

- สร้าง dataset สำหรับหลาย section
- สร้าง backup dataset
- สุ่มตรวจข้อความ
- ตัดข้อความไม่เหมาะสม
- สร้าง hidden test set
- สร้างชุดงานบ้านหรือแบบทดสอบปฏิบัติ

---

## 6. ข้อควรระวังด้านข้อมูล

ข้อความต้นทางเป็นเนื้อหาจาก social media จึงอาจมี:

- Emoji และสัญลักษณ์พิเศษ
- URL และ hashtag
- ภาษาไทยปนภาษาอังกฤษ
- การพิมพ์ซ้ำ เช่น “ดีมากกกก”
- ข้อความสั้นมาก
- คำหยาบหรือข้อความไม่เหมาะสม
- การประชดประชัน
- label ที่อาจคลุมเครือในบางข้อความ

### สิ่งที่ผู้สอนควรทำก่อนแจก

1. สุ่มตรวจอย่างน้อย 20–30 ข้อความต่อ label
2. Flag หรือตัดข้อความที่ไม่เหมาะกับห้องเรียน
3. ตรวจว่ามีข้อมูลส่วนบุคคลหรือไม่
4. ตรวจว่าจำนวน positive และ negative สมดุล
5. ตรวจว่า CSV เปิดด้วย UTF-8 ได้ถูกต้อง
6. ยืนยันว่า column names ตรงกับ Student Notebook

### แนวปฏิบัติในการใช้ Dataset

- ใช้เพื่อการเรียนการสอนภายในรายวิชา
- ไม่พยายามระบุตัวตนเจ้าของข้อความ
- ไม่ใช้ข้อความตัวอย่างเพื่อวิจารณ์บุคคล
- ไม่แชร์ข้อความที่มีความเสี่ยงด้านข้อมูลส่วนบุคคล
- ให้ credit dataset source ใน Notebook และเอกสารประกอบ

---

## 7. Pre-class Checklist

### ก่อนสอน 1–2 วัน

- [ ] ตรวจไฟล์ `customer_feedback_sentiment.csv`
- [ ] ตรวจว่ามี columns: `review_id`, `review_text`, `sentiment`, `source`
- [ ] ตรวจว่ามี labels: `positive`, `negative`
- [ ] ตรวจว่า label ทั้งสอง class สมดุล
- [ ] ตรวจ missing values และ duplicate rows
- [ ] ตรวจ encoding ของ CSV เป็น UTF-8 หรือ UTF-8-SIG
- [ ] ตรวจข้อความไม่เหมาะสมด้วยการสุ่มอ่าน
- [ ] ตรวจว่า Student Notebook ไม่มีเฉลยหลงเหลืออยู่
- [ ] ตรวจว่า Solution Notebook รันได้ครบ
- [ ] สร้าง ZIP สำรองของโฟลเดอร์ `student/`
- [ ] อัปโหลดไฟล์ไป Google Drive หรือ LMS
- [ ] ตั้งสิทธิ์ student folder เป็น Viewer
- [ ] เก็บ solution และ master dataset เป็น private

### ก่อนเริ่มคาบ

- [ ] เปิด Google Colab และลงชื่อเข้าใช้
- [ ] เปิด Student Notebook ผ่านลิงก์ในฐานะ Viewer
- [ ] ทดสอบ `File > Save a copy in Drive`
- [ ] ทดสอบ upload ไฟล์ CSV
- [ ] ทดสอบ run environment setup
- [ ] ทดสอบ run ทุก cell แบบ top-to-bottom
- [ ] เตรียมลิงก์สำรอง เช่น ZIP file หรือ Google Drive สำรอง
- [ ] เตรียม Solution Notebook สำหรับสาธิต
- [ ] แจ้งชื่อไฟล์ที่ต้องส่ง
- [ ] แจ้งเวลาส่งงานและช่องทางส่งงาน

---

## 8. Run-all Test

ก่อนแจก Notebook ต้องทดสอบด้วยขั้นตอนนี้:

1. เปิด Notebook ใหม่ใน Google Colab
2. เลือก:

   ```text
   Runtime → Disconnect and delete runtime
   ```

3. เปิด Notebook ใหม่
4. Upload `customer_feedback_sentiment.csv`
5. เลือก:

   ```text
   Runtime → Run all
   ```

6. ตรวจว่าไม่มี error
7. ตรวจว่า output หลักปรากฏครบ:
   - Data exploration
   - `clean_text`
   - Train/test split
   - Experiment unigram
   - Experiment bigram
   - Classification report
   - Confusion matrix
   - Error analysis table

### เกณฑ์ผ่าน

- รันทั้ง notebook ได้ภายในประมาณ 3–5 นาที
- ไม่มี dependency ที่ติดตั้งไม่สำเร็จ
- ไม่มี hard-coded path ของผู้สอน
- ไม่ต้องใช้ API key
- ไม่ต้องเชื่อม Google Drive เพื่อรันส่วนหลัก
- Error messages ที่เกิดจาก upload ไฟล์ผิดควรอ่านเข้าใจง่าย

---

## 9. แผนสอน 90 นาที

| เวลา | กิจกรรมผู้สอน | สิ่งที่นักศึกษาต้องได้ |
|---|---|---|
| 0–5 นาที | อธิบายโจทย์ ผลลัพธ์ และวิธีส่งงาน | เปิด notebook และ save copy |
| 5–10 นาที | แนะนำการ upload CSV และ run setup | โหลด dataset สำเร็จ |
| 10–20 นาที | อธิบาย dataset, labels และ data quality | ทำ Data Exploration |
| 20–35 นาที | สาธิต text cleaning และชี้ข้อควรระวัง | สร้าง `clean_text` |
| 35–45 นาที | อธิบาย train/test split และ data leakage | ได้ X_train/X_test |
| 45–60 นาที | อธิบาย TF-IDF, n-gram และ Logistic Regression | Train Experiment 1 |
| 60–70 นาที | ให้ทดลอง unigram + bigram | Train Experiment 2 และเทียบผล |
| 70–80 นาที | อธิบาย metrics และ confusion matrix | เลือก best model |
| 80–87 นาที | ให้วิเคราะห์ error 3 ตัวอย่าง | ได้ Error Analysis |
| 87–90 นาที | สรุป business insight และส่งงาน | Notebook พร้อมส่ง |

---

## 10. Teaching Checkpoints

### Checkpoint 1 — นาทีที่ 15

นักศึกษาควรมี:

- Dataset upload สำเร็จ
- DataFrame แสดงผลได้
- Data exploration output
- Class distribution plot
- คำตอบ Task 1 อย่างน้อยบางส่วน

### Checkpoint 2 — นาทีที่ 35

นักศึกษาควรมี:

- คอลัมน์ `clean_text`
- ตัวอย่าง raw text และ clean text
- คำตอบเรื่อง URL, emoji, punctuation และ negation

### Checkpoint 3 — นาทีที่ 55

นักศึกษาควรมี:

- `X_train`, `X_test`, `y_train`, `y_test`
- Train/test split สำเร็จ
- Experiment unigram รันสำเร็จ
- มีค่า Accuracy และ F1-score

### Checkpoint 4 — นาทีที่ 70

นักศึกษาควรมี:

- Experiment bigram รันสำเร็จ
- ตารางเปรียบเทียบ 2 experiments
- เลือก best model แล้ว

### Checkpoint 5 — นาทีที่ 85

นักศึกษาควรมี:

- Classification report
- Confusion matrix
- Error analysis อย่างน้อย 3 ข้อความ
- เริ่มเขียน business insight

---

## 11. สาระสำคัญที่ผู้สอนควรอธิบาย

### 11.1 Raw Text และ Clean Text

Raw text มักมี noise เช่น:

```text
ดีมากกกก!!! 😍😍
ดูรายละเอียด [https://example.com](https://example.com)
```

Clean text อาจเหลือ:

```text
ดีมากกกก
```

ประเด็นที่ควรชี้ให้เห็น:

- Cleaning ช่วยลด noise
- Cleaning ที่แรงเกินไปอาจทำให้ sentiment signal หาย
- Emoji, punctuation และการพิมพ์ซ้ำอาจมีความหมาย
- คำปฏิเสธ เช่น `ไม่`, `ไม่ได้`, `not` ไม่ควรถูกลบทิ้งแบบไม่พิจารณา

### 11.2 TF-IDF

อธิบายอย่างย่อ:

- TF = คำนี้ปรากฏมากเพียงใดในข้อความหนึ่ง
- IDF = คำนี้หายาก/จำเพาะเพียงใดเมื่อเทียบกับข้อความทั้งหมด
- TF-IDF ให้ค่าน้ำหนักสูงกับคำที่สำคัญในบางเอกสาร
- คำที่อยู่ในเกือบทุกข้อความมักมีน้ำหนักน้อยลง

### 11.3 N-grams

| รูปแบบ | ตัวอย่าง |
|---|---|
| Unigram | `ดี`, `แย่`, `ช้า` |
| Bigram | `ไม่ ดี`, `ส่ง ช้า`, `บริการ แย่` |

Bigram อาจช่วยจับวลีหรือคำที่อยู่ร่วมกัน แต่ทำให้จำนวน features เพิ่มขึ้น

### 11.4 Train/Test Split และ Data Leakage

เน้นประโยคนี้:

> ต้องแบ่ง Train/Test ก่อน fit TF-IDF Vectorizer

เหตุผล:

- TF-IDF เรียนรู้ vocabulary และ IDF จากข้อมูล
- หาก fit จากข้อมูลทั้งหมด test set จะรั่วเข้าไปในขั้นตอน train
- Pipeline ทำให้ vectorizer fit บน training set เมื่อเรียก `model.fit(X_train, y_train)`

### 11.5 Metrics

| Metric | ความหมาย |
|---|---|
| Accuracy | ทำนายถูกทั้งหมดกี่สัดส่วน |
| Precision Negative | ทายว่า negative แล้วเป็น negative จริงมากแค่ไหน |
| Recall Negative | negative จริงถูกตรวจพบมากแค่ไหน |
| F1 Negative | ค่ากลางระหว่าง precision และ recall |
| False Negative | negative จริง แต่โมเดลทาย positive |
| False Positive | positive จริง แต่โมเดลทาย negative |

### 11.6 Use Case

หากเป้าหมายคือไม่พลาดลูกค้าที่ไม่พอใจ:

```text
ให้ความสำคัญกับ Recall ของ class negative
```

แต่ Recall สูงเกินไปอาจทำให้:

```text
False Positive เพิ่มขึ้น
ทีม support ต้องตรวจข้อความมากขึ้น
```

---

## 12. Expected Results

### ผลที่ควรคาดหวัง

ผลลัพธ์จริงขึ้นกับ:

- dataset version
- random state
- preprocessing rules
- package version
- การลบ duplicate
- parameter ของ TF-IDF

ดังนั้นไม่ควรกำหนด Accuracy/F1-score แบบตายตัว

### สิ่งที่ควรเกิดขึ้นโดยทั่วไป

- Dataset มี 800 ข้อความก่อน cleaning
- หลัง cleaning จำนวนอาจลดลงเล็กน้อย
- Train set ประมาณ 80%
- Test set ประมาณ 20%
- Vocabulary ของ bigram มักมากกว่า unigram
- Bigram อาจช่วยหรือไม่ช่วย F1-score ก็ได้
- Logistic Regression ควร train ได้ภายในเวลาอันสั้น
- มีข้อความบางส่วนที่โมเดลทำนายผิด
- นักศึกษาควรสามารถอธิบาย error ได้มากกว่าบอกเพียงว่า “โมเดลไม่แม่น”

### คำตอบที่ยอมรับได้

- เลือก unigram หรือ bigram เป็น best model ได้
- ต้องมีเหตุผลจาก metrics โดยเฉพาะ F1-score negative
- Error analysis ไม่จำเป็นต้องเหมือนเฉลยทุกคำ
- Business insight ต้องเชื่อมกับผลการทดลองจริง

---

## 13. Troubleshooting Guide

### ปัญหา: `FileNotFoundError`

อาการ:

```text
ไม่พบไฟล์ customer_feedback_sentiment.csv
```

สาเหตุ:

- นักศึกษายังไม่ได้ upload ไฟล์
- ชื่อไฟล์ไม่ตรง
- Runtime reset แล้วไฟล์ใน Colab หาย

แนวทางแก้:

1. รัน upload cell ใหม่
2. ตรวจว่าชื่อไฟล์ตรงกับ `DATA_FILE`
3. ใช้คำสั่ง:

```python
!ls -lah
```

4. ตรวจชื่อไฟล์ที่อยู่ใน `/content`

---

### ปัญหา: `KeyError: 'review_text'`

สาเหตุ:

- CSV ไม่มี column ชื่อ `review_text`
- ชื่อ column สะกดผิด
- ใช้ dataset คนละไฟล์

แนวทางแก้:

```python
print(df.columns.tolist())
```

ชื่อที่ต้องมี:

```text
review_id
review_text
sentiment
source
```

---

### ปัญหา: `KeyError: "['char_count'] not in index"`

สาเหตุ:

- code เลือก column `char_count`
- แต่ `char_count` ถูกลบไปแล้วก่อนหน้า

แนวทางแก้:

```python
print(full_df.columns.tolist())
```

หากต้องการเก็บ `char_count`:

```python
# อย่าลบบรรทัดนี้ใน prepare_master_dataset
df["char_count"] = df["review_text"].str.len()
```

หากไม่ต้องการเก็บ:

```python
# ลบ "char_count" ออกจากรายชื่อ columns ตอนเลือก full_df
```

---

### ปัญหา: `ValueError: The test_size should be greater or equal to the number of classes`

สาเหตุ:

- Dataset เล็กเกินไป
- มีเพียง class เดียวหลัง cleaning
- แบ่งข้อมูลด้วย `stratify=y` แต่ตัวอย่างต่อ class น้อยเกินไป

แนวทางแก้:

```python
display(df["sentiment"].value_counts())
```

ตรวจให้แน่ใจว่ามีทั้ง:

```text
positive
negative
```

---

### ปัญหา: `ValueError: empty vocabulary`

สาเหตุที่เป็นไปได้:

- Text cleaning ลบข้อความออกเกือบหมด
- ตั้ง `min_df` สูงเกินไป
- ข้อความว่างทั้งหมด
- ใช้ stopwords มากเกินไป

แนวทางแก้:

```python
display(df["clean_text"].head(20))
print(df["clean_text"].str.len().describe())
```

จากนั้น:

```python
min_df = 1
```

และตรวจว่า `clean_text` ยังมีข้อความเหลืออยู่

---

### ปัญหา: Logistic Regression ไม่ converge

อาการ:

```text
ConvergenceWarning
```

แนวทางแก้:

```python
LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)
```

สำหรับ dataset 800 แถว `max_iter=1000` โดยทั่วไปควรเพียงพอ

---

### ปัญหา: ไม่มีข้อความทำนายผิด

สาเหตุ:

- Dataset ง่ายเกินไป
- Train/test leakage
- test set เล็กเกินไป
- ข้อมูลมี duplicates ระหว่าง train/test

ตรวจ:

```python
set(X_train).intersection(set(X_test))
```

ผลลัพธ์ควรเป็น:

```python
set()
```

หากต้องการให้มี error สำหรับวิเคราะห์:

- ใช้ข้อความที่หลากหลายขึ้น
- ลดการคัดกรองข้อมูลมากเกินไป
- เพิ่มจำนวนข้อมูล
- ไม่ใช้ dataset ที่มีคำบ่งชี้ label ชัดเจนเกินไป

---

### ปัญหา: UTF-8 อ่านภาษาไทยผิด

ลองโหลดไฟล์ด้วย:

```python
df = pd.read_csv(
    "customer_feedback_sentiment.csv",
    encoding="utf-8-sig"
)
```

แนะนำให้บันทึกไฟล์ด้วย:

```python
df.to_csv(
    "customer_feedback_sentiment.csv",
    index=False,
    encoding="utf-8-sig"
)
```

---

## 14. แนวทางตรวจงานอย่างรวดเร็ว

### ต้องมีอย่างน้อย

- [ ] ข้อมูลนักศึกษาครบ
- [ ] Data exploration output
- [ ] คำตอบ Task 1
- [ ] คอลัมน์ `clean_text`
- [ ] Train/test split พร้อม `stratify=y`
- [ ] ผล unigram หรือ baseline model
- [ ] ผล bigram หรือการทดลอง parameter อีก 1 แบบ
- [ ] ตารางเปรียบเทียบผล
- [ ] Classification report
- [ ] Confusion matrix
- [ ] Error analysis อย่างน้อย 3 ข้อความ
- [ ] Business insight

### Rubric

| หัวข้อ | คะแนน | หลักฐาน |
|---|---:|---|
| Data Exploration | 10 | shape, missing, duplicate, class distribution |
| Text Cleaning | 20 | `clean_text` และคำอธิบายผล cleaning |
| Train/Test Split | 10 | `test_size=0.20`, `stratify=y`, อธิบาย leakage |
| TF-IDF + Model | 20 | Pipeline และผลทดลองอย่างน้อย 2 configurations |
| Evaluation | 20 | classification report และ confusion matrix |
| Error Analysis | 10 | วิเคราะห์อย่างน้อย 3 ข้อความ |
| Business Insight | 10 | เชื่อมผลกับ customer support use case |
| **รวม** | **100** |  |

---

## 15. เฉลยแนวคิดที่ผู้สอนใช้ตรวจ

### คำถาม: ทำไมต้องใช้ `stratify=y`

แนวคำตอบ:

> เพื่อให้สัดส่วน positive และ negative ในชุด train/test ใกล้เคียงกับข้อมูลต้นฉบับ ลดความเสี่ยงที่ test set มี class ใด class หนึ่งน้อยเกินไป

### คำถาม: Data leakage คืออะไร

แนวคำตอบ:

> การที่ข้อมูลจาก test set มีอิทธิพลต่อการ train เช่น fit TF-IDF vocabulary หรือ IDF จากข้อมูลทั้งหมดก่อนแบ่ง train/test ทำให้ผล evaluation ดีเกินจริง

### คำถาม: ทำไม Recall Negative สำคัญ

แนวคำตอบ:

> เพราะช่วยวัดว่าโมเดลตรวจพบข้อความ negative จริงได้มากเพียงใด หาก Recall ต่ำ บริษัทอาจพลาด feedback ของลูกค้าที่ไม่พอใจ

### คำถาม: ทำไม TF-IDF + Logistic Regression ยังมีข้อจำกัด

แนวคำตอบ:

> ไม่เข้าใจบริบทเชิงลึก ลำดับคำที่ซับซ้อน การประชดประชัน คำปฏิเสธ และความสัมพันธ์เชิง semantic ของคำได้ดีเท่าโมเดล embedding หรือ Transformer

### คำถาม: Bigram มีประโยชน์อย่างไร

แนวคำตอบ:

> ช่วยให้โมเดลพิจารณาคำที่อยู่ติดกัน เช่น “ไม่ ดี” หรือ “ส่ง ช้า” ซึ่งอาจมีความหมายชัดกว่าดูเป็นคำเดี่ยว

---

## 16. กิจกรรมเสริมหลังแลป

สำหรับนักศึกษาที่ทำเสร็จเร็ว หรือใช้เป็น homework:

1. เปรียบเทียบ `MultinomialNB` กับ Logistic Regression
2. ปรับ `min_df` เป็น 1, 2 และ 5
3. ปรับ `ngram_range` เป็น `(1, 1)`, `(1, 2)` และ `(1, 3)`
4. เพิ่ม stopword handling
5. ทดลอง Thai tokenization ด้วย PyThaiNLP
6. ทดลอง character-level TF-IDF
7. ดู top predictive terms ของแต่ละ class
8. วิเคราะห์ topic ด้วย LDA
9. ทดลอง text clustering ด้วย K-Means
10. ทดลอง transformer สำหรับภาษาไทยในบทถัดไป

---

## 17. ข้อความสรุปท้ายคาบ

ใช้สรุปให้นักศึกษา:

> ในแลปนี้เราเปลี่ยนข้อความดิบให้เป็น features ด้วย TF-IDF  
> จากนั้นใช้ Logistic Regression เพื่อจำแนก sentiment  
> สิ่งสำคัญไม่ได้อยู่ที่ Accuracy เพียงอย่างเดียว แต่คือการเข้าใจ
> Precision, Recall, F1-score และข้อผิดพลาดของโมเดล  
> สำหรับงานจริง เราต้องเชื่อมผลของโมเดลกับความเสี่ยงและต้นทุนทางธุรกิจด้วย

---

## 18. References

- Wisesight Sentiment Corpus / PyThaiNLP
- scikit-learn: `TfidfVectorizer`
- scikit-learn: `Pipeline`
- scikit-learn: `LogisticRegression`
- scikit-learn: `classification_report`
- scikit-learn: `ConfusionMatrixDisplay`
- Google Colab Documentation
