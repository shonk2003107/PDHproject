# PDHproject

פרויקט גמר: סימולציה של נעילת לייזר למהוד Fabry-Perot בשיטת Pound-Drever-Hall (PDH) —
התאמת מודים (Mode Matching), יציבות ותגובת המהוד, וגילוי הומודיין עם רעש פאזה.

## מבנה הפרויקט

```
PDHproject/
├── src/                        # ליבה פיזיקלית/מתמטית בלבד (ללא גרפים)
│   ├── optics_utils.py         # מטריצות ABCD, פרמטר q, חפיפת מודים, אופטימיזציית שתי עדשות
│   ├── fabry_perot.py          # מחלקת FabryPerot: FSR, Finesse, יציבות, מצב עצמי, החזרה/מעבר, אות שגיאה PDH
│   └── signal_dsp.py           # רעש פאזה (Wiener), הומודיין מאוזן, רעש שוט, ערבול, LPF, PSD
├── simulations/                # סקריפטים להרצה
│   ├── sim1_mode_matching.py   # בחירת זוג עדשות ומיקומן להתאמה למוד המהוד + סבולות
│   ├── sim2_cavity_stability.py# דיאגרמת יציבות, waist ו-TMS מול L, תגובת המהוד ואות PDH
│   └── sim3_homodyne_noise.py  # הומודיין עם רעש פאזה ורעש שוט, דמודולציה והשוואה לתאוריה
├── data_out/                   # תוצרי הסימולציות (png/csv) — לא נשמר בגיט
└── requirements.txt
```

## התקנה

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## הרצה

מתיקיית השורש של הפרויקט:

```bash
python simulations/sim1_mode_matching.py
python simulations/sim2_cavity_stability.py
python simulations/sim3_homodyne_noise.py --show   # --show פותח את חלון הגרף
```

הגרפים וקבצי ה-CSV נשמרים ב-`data_out/`. הפרמטרים של כל סימולציה מוגדרים בראש הקובץ שלה.
