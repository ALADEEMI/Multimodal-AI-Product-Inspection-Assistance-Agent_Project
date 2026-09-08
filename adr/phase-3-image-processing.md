# Phase 3 — معالجة الصور وتوثيقها

## Pipeline

قراءة RGB → resize موحد مع mask مطابق → denoise خفيف → contrast enhancement موثق → `float32` normalization بإحصاءات train → augmentation للتدريب فقط. استخدم nearest-neighbor دائماً للـmask.

## التنفيذ

أنشئ `vision/preprocessing.py`, `image_processing.py`, `dataset.py` وعقد `preprocess(image, mask, stage) -> image_tensor,mask_tensor,metadata`. طبق flip/rotate/brightness على train فقط، والتحويل الهندسي نفسه على الصورة والقناع. احفظ raw/denoise/contrast/overlay لعينات `good` وكل defect في `docs/visuals/preprocessing/`.

## تحقق

اختبر أن القناع ثنائي ومحاذٍ بعد التحويل، وأن `eval` حتمي، وأن validation/test لا يستدعيان augmentation. احفظ mean/std من train فقط مع الإصدار. Edge detection اختياري للتقرير فقط ولا يدخل النموذج بلا تجربة مقارنة.

**بوابة الخروج:** DataLoader يطابق أبعاد image/mask، والإحصاءات والتوثيق البصري محفوظان.
