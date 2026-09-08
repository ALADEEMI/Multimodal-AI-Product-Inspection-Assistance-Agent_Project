# Phase 5 — Information Extraction والسياسة المحلية

## النطاق

تحويل مخرجات التحليل إلى حقائق منظمة بلا LLM، ثم lookup محلي مشروط. لا تعتبر السياسة المبدئية في CSV سياسة تجارية حقيقية؛ تستبدل بالنص المعتمد قبل العرض النهائي.

## العقد

```json
{"Product":"bottle","Problem":"Damage","Severity":"High",
 "Requested_Action":"replacement","evidence":{"text_confidence":0.0}}
```

`Requested_Action ∈ {return,replacement,repair,unspecified}`. القواعد تبحث بالعربية والإنجليزية عن مرادفات الإرجاع والاستبدال والإصلاح مع أولوية موثقة عندما تتعارض.

## التنفيذ والاختبار

أنشئ `extraction/information_extraction.py` و`tools/policy_lookup.py`. validator لـ`data/policies.csv` يطلب `product_category,return_window_days,warranty_summary`، ويرفض أياماً سالبة أو فئة مكررة. لا يستدعى tool إلا لـreturn/replacement؛ no-match يعيد حالة واضحة لا نصاً مخترعاً. اختبر كل action، نصاً بلا action، فئة لا سياسة لها، وحالة تعارض كلمات.

**بوابة الخروج:** extraction deterministic، policy lookup محلي ومختبر، والـCSV معتمد/موسوم بأنه Demo.
