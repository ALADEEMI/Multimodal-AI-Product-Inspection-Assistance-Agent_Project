# Phase 9 — التوثيق والتسليم

## مخرجات التسليم

1. `docs/nlp_report.md`: Dirty Data، preprocessing، BiLSTM، splits، metrics وConfusion Matrix.
2. `docs/vision_dl_report.md`: قبل/بعد المعالجة، U-Net، loss، IoU/Dice/Pixel Accuracy، masks مرئية.
3. `docs/agent_report.md`: Graph، state، tool use المحلي، prompt safety، Fusion comparison والقيود.
4. `README.md`: متطلبات البيئة، وضع البيانات، أوامر التوليد/التدريب/التشغيل، الترخيص، وعدم تضمين الأسرار.

## قائمة مراجعة نهائية

- Seed/config/data/model versions مذكورة بجانب كل metric.
- النتائج من test غير المستخدم في الاختيار، ومصادر الصور وترخيصها ظاهرة.
- dataset synthetic موصوف صراحة مع حدوده.
- جدول Fusion لا يدعي التحسن بلا دليل.
- Demo checklist يثبت النص/الصورة/كليهما/no-defect، وfallback LLM.
- سياسة CSV وسمت Demo أو استبدلت بسياسة معتمدة.

**بوابة التسليم:** عضو فريق آخر يستطيع إعادة إعداد البيئة وتشغيل dataset→models→demo من README فقط، وتطابق artifacts التقارير.
