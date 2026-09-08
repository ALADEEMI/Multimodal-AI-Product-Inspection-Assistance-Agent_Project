# Phase 2 — NLP وBiLSTM متعدد المهام

## البنية والعقد

`Embedding → Shared BiLSTM → Dropout → {Sigmoid is_complaint, Softmax problem_type}`. الملفات: `cleaning.py`, `preprocessing.py`, `model.py`, `train_rnn.py`, `inference.py`, `evaluation/nlp_metrics.py`.

## التنفيذ

1. احتفظ بالنص الخام؛ طبّع Unicode والمسافات فقط ولا تمح إشارات Dirty Data إلا بتجربة موثقة.
2. fit للـtokenizer وlabel-map على train فقط؛ أضف OOV وpadding بطول محدد من train.
3. درب loss مركب `w_binary*BCE + w_multi*CE` وسجل كل hyperparameter والـseed.
4. اختر checkpoint بأفضل validation macro-F1، لا بنتائج test.
5. احفظ model/tokenizer/label-map/config/metrics، وأتح `get_text_embedding` من الطبقة قبل الرأس.

## اختبار وتقييم

للثنائي: Accuracy/Precision/Recall/F1. وللمتعدد: Accuracy/Macro-F1/per-class F1/Confusion Matrix. inference يعيد `is_complaint`, confidences, `problem_type`, و`float32[TEXT_EMBED_DIM]`. اختبر نصاً فارغاً، عدم إعادة fit للـtokenizer، وثبات بعد embedding.

**بوابة الخروج:** artifacts تعاد في عملية نظيفة، metrics محفوظة، وواجهة embedding مستقرة.
