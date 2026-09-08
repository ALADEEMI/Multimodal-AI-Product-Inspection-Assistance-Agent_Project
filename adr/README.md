# وثائق التنفيذ حسب المرحلة

هذا المجلد يحول قرار [ADR-001](../docs/ADR-001-multimodal-inspection-architecture-and-delivery-plan.md) إلى تعليمات تنفيذ قابلة للمراجعة. كل ملف يحدد النطاق والواجهات والخطوات والاختبارات وArtifacts وبوابة الانتقال.

| Phase | الوثيقة | بوابة البدء |
|---|---|---|
| 0 | [التأسيس](phase-0-foundation.md) | توفر `bottle` محلياً |
| 1 | [البيانات المقترنة](phase-1-paired-dataset.md) | نجاح validator |
| 2 | [NLP](phase-2-nlp.md) | Manifest صالح |
| 3 | [معالجة الصور](phase-3-image-processing.md) | بيانات خام صالحة |
| 4 | [التقسيم](phase-4-segmentation.md) | Pipeline الصور |
| 5 | [الاستخراج والسياسة](phase-5-extraction-policy.md) | artifacts NLP |
| 6 | [Fusion](phase-6-fusion.md) | embeddings النموذجين |
| 7 | [Agent](phase-7-agent.md) | Fusion وpolicy tool |
| 8 | [Demo/E2E](phase-8-demo-e2e.md) | Graph مختبر |
| 9 | [التسليم](phase-9-delivery.md) | نتائج E2E |

لا يدمج أي Slice إلا باختبار موثق، ولا تُستخدم بيانات الاختبار في ضبط النموذج.
