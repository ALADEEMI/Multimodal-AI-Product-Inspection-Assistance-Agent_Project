# موضع بيانات MVTec في بيئة العمل الحالية

جذر النسخة المنزلة هو `data/raw/mvtec/bottle/`، والفئة المختارة تقع في:

```text
data/raw/mvtec/bottle/bottle/
├── train/good/
├── test/{good,broken_large,broken_small,contamination}/
└── ground_truth/{broken_large,broken_small,contamination}/
```

هذه التسمية المتداخلة سببها أن المصدر العام يحتوي جميع الفئات داخل مجلد استنساخ سُمّي `bottle`. قبل تنفيذ Phase 0، يضبط الإعداد هذا المسار الفعلي بدلاً من افتراض اسم مجلد فقط. لا تنقل أو تحذف أي بيانات الآن؛ ترتيبها اختياري بعد نجاح الـvalidator.
