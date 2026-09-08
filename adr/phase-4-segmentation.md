# Phase 4 — Deep Learning للتقسيم

## النطاق

تدريب U-Net مصغر supervised على الصور المعيبة وماسكات MVTec؛ القناع المتوقع دليل موضع العيب. لا تستخدم مجموعة الاختبار لاختيار البنية أو epoch.

## التنفيذ

1. أنشئ `vision/model.py`, `train_cnn.py`, `inference.py`, `evaluation/vision_metrics.py`.
2. DataLoader يقرأ العيوب والماسكات المتطابقة؛ عالج `good` بسياسة مكتوبة (استبعاد تدريب القناع أو inclusion بقناع صفري) واستخدمها في جميع التجارب.
3. ابدأ بـBCE+Dice loss، Adam، early stopping على validation Dice، وcheckpoint لأفضل validation فقط.
4. سجل حجم الإدخال، القنوات، base filters، learning rate، batch، epochs، augmentations، seed، ونسبة defect pixels.
5. inference يعيد `mask_probability`, `binary_mask`, `segmentation_confidence`, `location_summary`, و`float32[IMAGE_EMBED_DIM]` من encoder/pooling معرف.

## التقييم

احسب IoU وDice وPixel Accuracy على test مرة واحدة بعد التجميد. احفظ predicted/ground-truth/overlay لعينات متنوعة. اختبر شكل mask، threshold ثابتاً، وتحميل checkpoint دون training code.

**بوابة الخروج:** تقرير metrics وعينات مرئية وimage embedding بعقد ثابت للـFusion.
