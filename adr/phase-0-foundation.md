# Phase 0 — التأسيس والتشغيل القابل لإعادة الإنتاج

## النطاق

أنشئ البنية المشتركة بلا تدريب. الفئة المعتمدة `bottle` في `data/raw/mvtec/bottle`. لا تُعدّل البيانات الخام ولا تُرفع إلى Git.

## المخرجات

```text
config.py, requirements.txt, .gitignore, scripts/validate_mvtec.py
scripts/create_manifest.py, tests/, logs/, artifacts/{nlp,vision,fusion}/
nlp/, vision/, text_generator/, extraction/, fusion/, agent/, evaluation/, app/
```

## التنفيذ

1. `config.py` يقرأ `PROJECT_SEED` و`MVTEC_CATEGORY` من البيئة ويعيد مسارات نسبية فقط.
2. ثبت Python موحداً وحزم PyTorch/OpenCV/pandas/scikit-learn/LangGraph/pytest وواجهة demo بإصدارات مقيدة.
3. validator يتحقق من `train/good`, `test/*`, `ground_truth/*` وتطابق صورة العيب مع `<stem>_mask.png`.
4. manifest يسجل `sample_id,image_path,split_source,defect_type,is_defective,mask_path` بلا نص مولّد.
5. seed موحد لـrandom/numpy/torch؛ سجل الإعدادات ونسخ الحزم في `logs/run_metadata.json`.

## العقد والاختبارات

`validate_mvtec(path) -> {valid, counts_by_split, missing_masks, errors}` وينتهي برمز غير صفري عند الفشل. اختبر بنية مؤقتة سليمة وأخرى بقناع مفقود؛ ثم تحقق من تفرد `sample_id` وقابلية قراءة كل المسارات.

**بوابة الخروج:** البيئة تثبت، validator ناجح، manifest محفوظ، ولا أسرار أو بيانات خام tracked.
