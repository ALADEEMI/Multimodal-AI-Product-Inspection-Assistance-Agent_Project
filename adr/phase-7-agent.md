# Phase 7 — LangGraph Agent

## State ومسار الرسم

`START → receive → route_modalities → text_analyzer/image_analyzer → extraction → fusion → context → conditional_policy → llm → END`.

State قابل لـserialization ويشمل `raw_text`, `image_ref`, outputs النماذج، extracted fields، fusion، policy، `errors`, و`trace_id`. يمنع حفظ API key أو image bytes الكبيرة في trace.

## التنفيذ

1. `receive` يرفض غياب النص والصورة ويحدد حالة كل modality.
2. analyzers لا تستدعي LLM، وتعيد fallback موثقاً عند artifact مفقود أو confidence منخفض.
3. `context_builder` يمرر أدلة فقط: labels، confidences، location summary، uncertainty، وpolicy المحلية إن وجدت.
4. شرط السياسة يتحقق من `return/replacement` فقط.
5. LLM adapter يقرأ `.env`، يتحقق من توفره، ويعيد تقريراً منظماً؛ عند الفشل يعيد template من facts بدلاً من توقف graph.

## Prompt وضمان الجودة

ينص prompt: لا تخترع منتجاً أو عيباً أو سياسة؛ اذكر عدم اليقين؛ لا تعط نصيحة سلامة/ضمان خارج context. اختبر النص فقط والصورة فقط وكليهما ومدخل فارغ وفشل LLM وno-policy. احفظ trace مختصراً لكل سيناريو.

**بوابة الخروج:** جميع المسارات تنتهي برد منظم، والسياسة لا تستدعى خارج شرطها.
