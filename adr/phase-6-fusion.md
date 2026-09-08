# Phase 6 — Fusion متعدد الوسائط

## الهدف

إثبات أو نفي قيمة دمج embeddings بإنصاف. لا يكفي عرض نتيجتي نموذجين بجانب بعضهما.

## التنفيذ

1. أنشئ manifest embeddings يتضمن `sample_id`, split، vector text، vector image، modality availability، والوسم المستهدف؛ يرفض عدم تطابق المعرفات.
2. baseline: `concat(text_embedding, image_embedding, modality_mask) → Dense/ReLU/Dropout → output head`.
3. طبّق modality masks وصفر vector الغائب لتدعم text-only/image-only/both بلا تغيير أبعاد.
4. جمد encoders أولاً؛ لا تعدلهما في نفس تجربة Fusion. استخدم train/validation/test نفسها ومقاييس هدف موحدة.
5. احفظ `fusion_model`, scalers/config, feature dimensions, checkpoint, metrics وper-sample predictions.

## التقييم

اعرض جدول النص مقابل الصورة مقابل Fusion على test نفسه مع support وفواصل/seed إن أمكن. حلل حالات النجاح والإخفاق؛ إن لم يتحسن Fusion، سجل ذلك بوضوح ولا تعدل test حتى يظهر تحسن.

**بوابة الخروج:** نموذج قابل للتحميل، جدول مقارنة قابل للإعادة، وتحليل صادق للنتيجة.
