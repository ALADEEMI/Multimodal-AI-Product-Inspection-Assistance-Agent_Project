# Phase 1 — Dataset النص والصورة المقترنة

## النطاق

أنشئ dataset قابل لإعادة الإنتاج من manifest؛ لا تستخدم LLM لتوليد النص. كل نص يأتي من templates versioned وتنويع RNG محلي.

| defect_type | is_complaint | problem_type | severity |
|---|---:|---|---|
| `good` | false | `Other` | `Low` |
| `broken_large` | true | `Damage` | `High` |
| `broken_small` | true | `Damage` | `Medium` |
| `contamination` | true | `Quality` | `Medium` |

## التنفيذ

1. أنشئ `templates.py`, `noise_injection.py`, `generator.py`, `validate_dataset.py`.
2. وفر 5 templates لكل شكوى و4 نصوص سليمة على الأقل. لا تخترع موقع عيب دقيقاً.
3. اشتق seed النص من `sample_id`؛ طبق أخطاء/إيموجي/تكرار/حذف ترقيم بنسبة قابلة للضبط.
4. احسب تعديل الشدة من مساحة mask النسبية فقط، وحافظ على split قبل أي augmentation.
5. أخرج `paired_dataset.csv` بالعقد: `sample_id,image_path,generated_text,is_complaint,problem_type,defect_type,severity,mask_path,split,generator_version,seed`.

## الجودة

validator يرفض تكرار المعرفات، المسارات المكسورة، قناعاً لسليم، عيباً بلا قناع، وenum غير صالح. شغّل المولد مرتين بالـseed نفسه وتحقق من تطابق الحقول النصية. راجع 20 صفاً عشوائياً وسجل جدول counts حسب split/problem/defect في `dataset_card.md`.

**بوابة الخروج:** لا تسرب بين splits، CSV صالح، ومراجعة 20 نصاً موثقة.
