# PRD — Multimodal AI Product Inspection & Assistance Agent

**الحالة:** معتمد للبدء بالتنفيذ
**النطاق:** مشروع أكاديمي يخدم 3 مواد (NLP، Image Processing، Deep Learning) + طبقة AI Agent
**المرجع الكامل للنقاش:** `project-discussion-summary.md`

---

## 1. نظرة عامة (Overview)

بناء نظام Multimodal يستقبل نص شكوى و/أو صورة منتج، يحلّل كل مسار بموديل مختص (RNN للنص، CNN للصورة)، يدمج النتائج فعليًا (Fusion)، ثم يُخرج تقريرًا نهائيًا منظّمًا عبر LangGraph Agent + LLM. **لا يوجد أي استدعاء لخدمات API خارجية** في هذا الإصدار.

## 2. الأهداف (Goals)

- G1: بناء Dataset مدمجة (نص+صورة) متطابقة فعليًا، وليست مجمّعة قسريًا.
- G2: تدريب نموذج RNN متعدد المهام (Two-stage) بأداء موثّق (Accuracy/F1/Confusion Matrix).
- G3: تدريب نموذج CNN لتحديد موقع العيب (Segmentation) بأداء موثّق (IoU/Dice).
- G4: تحقيق Fusion حقيقي على مستوى التمثيلات (Embeddings) وإظهار أن الدمج يحسّن الثقة/الدقة مقارنة بكل نموذج منفرد.
- G5: بناء Agent عبر LangGraph ينسّق كامل الـPipeline وينتج تقريرًا نهائيًا عبر LLM بدون Hallucination.
- G6: توثيق يربط كل عنصر بمتطلبات كل مادة من المواد الثلاث.

## 3. خارج النطاق (Out of Scope)

- أي استدعاء API خارجي (RapidAPI أو غيره) — حُذف كمتطلب.
- دعم لغات غير العربية/الإنجليزية في النص المولّد ما لم يُطلب لاحقًا.
- Frontend إنتاجي كامل — تكفي واجهة تجريبية (Streamlit/Gradio).
- تغطية كل الـ15 فئة من MVTec دفعة واحدة (تبدأ بفئة واحدة، والتوسّع اختياري لاحقًا).

## 4. الفريق والقيود

- فريق من شخصين، عمل متوازي قدر الإمكان.
- لا توجد قيود على مدة التنفيذ في هذا الإصدار من الوثيقة (تُدار زمنيًا من الفريق نفسه).

---

## 5. متطلبات البيانات (Data Requirements)

### 5.1 مصدر الصور
- **MVTec AD** — الترخيص CC BY-NC-SA 4.0 (استخدام أكاديمي/غير تجاري فقط).
- البدء بفئة واحدة (مقترح: `bottle` أو `hazelnut`)، مع إبقاء الكود قابلًا للتوسعة لفئات إضافية بدون إعادة هيكلة.
- تحتوي كل فئة على: صور تدريب سليمة، صور اختبار (سليمة + معيبة)، وماسكات Ground Truth لمواقع العيوب في الصور المعيبة.

### 5.2 مولّد النصوص المطابقة (Text Generator) — عنصر جديد يُبنى من الصفر
**المدخل:** label الصورة من MVTec (نوع المنتج + نوع العيب + وجود/غياب عيب).
**المخرج:** نص شكوى شبيه بمراجعة عميل حقيقية، "متسخ" لغويًا بشكل متعمد.

**متطلبات التصميم:**
- بنك Templates متعدد لكل نوع عيب (لا يقل عن 4-5 صياغات مختلفة لكل نوع عيب لتفادي التكرار الممل).
- طبقة تنويع عشوائي (اختيار مرادفات، ترتيب جمل مختلف).
- طبقة "تلوّث" متعمد قابلة للضبط (نسبة مئوية): أخطاء إملائية، حروف مكررة، إيموجي، اختصارات عامية، غياب علامات ترقيم.
- توليد حقل `severity` (Low/Medium/High) مرتبط منطقيًا بنوع/حجم العيب.

**Schema المخرج النهائي (ملف CSV أو JSON موحّد):**

```
image_path        : str   (مسار الصورة الأصلية في MVTec)
generated_text     : str   (نص الشكوى المولّد)
is_complaint        : bool  (Label لرأس RNN الأول)
problem_type        : str   (Quality | Packaging | Price | Delivery | Damage | Other)
defect_type         : str   (من label الصورة الأصلي: scratch, crack, ...)
severity           : str   (Low | Medium | High)
mask_path          : str | null (مسار قناع العيب إن وُجد)
```

**معيار القبول (Definition of Done لهذه المرحلة):**
- كل صورة معيبة في الفئة المختارة لها سجل نصي مطابق واحد على الأقل.
- توزيع `problem_type` غير منحرف بشكل متطرف (لا فئة تمثل أكثر من ~50% من العيّنات دون مبرر).
- عيّنة عشوائية من 20 نصًا تمت مراجعتها يدويًا وتبدو منطقية ومرتبطة فعليًا بنوع العيب.

---

## 6. متطلبات وظيفية بالتفصيل (حسب المرحلة)

### المرحلة 0 — الإعداد
**المطلوب:**
- هيكلة المجلدات التالية:
```
project/
├── data/
│   ├── raw/mvtec/<category>/...
│   └── processed/paired_dataset.csv
├── nlp/  {cleaning.py, preprocessing.py, train_rnn.py, inference.py}
├── vision/  {preprocessing.py, image_processing.py, train_cnn.py, inference.py}
├── text_generator/  {templates.py, generator.py, noise_injection.py}
├── extraction/  {information_extraction.py}
├── fusion/  {fusion_model.py}
├── agent/  {graph.py, nodes.py, prompts.py, context.py}
├── evaluation/  {nlp_metrics.py, vision_metrics.py, fusion_metrics.py}
├── app/  {demo_ui.py}
└── docs/  {nlp_report.md, vision_dl_report.md, agent_report.md}
```
- بيئة Python موحّدة (requirements.txt واحد للفريقين لتفادي تعارض الإصدارات).
- تحميل MVTec AD (الفئة المختارة) ووضعها في `data/raw/mvtec/`.

### المرحلة 1 — مولّد النصوص (تفاصيل في القسم 5.2 أعلاه)

### المرحلة 2 — NLP Pipeline (RNN)
- Cleaning: إزالة/تطبيع الرموز غير القياسية، مع الإبقاء على إشارات "التلوّث" المقصودة للتدريب عليها (لا نزيل كل الضوضاء، فالهدف تعليم النموذج التعامل معها).
- Tokenization + Stopword Removal + Padding.
- بنية النموذج: Embedding → **Shared BiLSTM Encoder** (خوارزمية معتمدة نهائيًا) → رأسين (Dense+Sigmoid للـ`is_complaint`, Dense+Softmax للـ`problem_type`).
- Loss مركّب (مجموع موزون للرأسين).
- تقييم منفصل لكل رأس: Accuracy/F1 (رأس 1)، Accuracy/F1-macro + Confusion Matrix (رأس 2).
- حفظ: الموديل + الـTokenizer + ملف تقييم.

### المرحلة 3 — Image Processing Pipeline
- خطوات إلزامية: Resize، تقليل الضوضاء، تحسين التباين، Normalization، Augmentation (Flip/Rotate/Brightness).
- خطوة موثّقة بصريًا (Before/After) لكل عملية — تُحفظ كصور في `docs/` لدعم تقرير مادة Image Processing.
- (اختياري إن سمح الوقت) Edge Detection كتحليل استكشافي قبل التغذية للـCNN.

### المرحلة 4 — Deep Learning (CNN Segmentation)
- بنية مقترحة: Encoder-Decoder مبسّط (U-Net مصغّر) مناسب لحجم بيانات MVTec المحدود.
- تدريب Supervised مباشر على الصور المعيبة + ماسكاتها الرسمية.
- Loss: Dice Loss أو BCE + Dice مجتمعين.
- تقييم: IoU، Dice Score، Pixel Accuracy — على مجموعة اختبار منفصلة.
- حفظ الموديل + عيّنات تصويرية لمخرجات الـSegmentation (قناع متوقع مقابل الحقيقي).

### المرحلة 5 — Information Extraction
- مدخل: مخرجات RNN (problem_type, severity) + النص الأصلي.
- منطق Rule-based (وليس نموذج منفصل، لتوفير الوقت) يحوّل هذه المخرجات إلى حقول منظمة:
```
{Product, Problem, Severity, Requested_Action}
```
- `Requested_Action` يُستنتج من كلمات مفتاحية بالنص (استبدال/استرجاع/إصلاح...) إن وُجدت، وإلا "غير محدد".

### المرحلة 6 — Multimodal Fusion
- استخراج Feature Vector من الطبقة قبل الأخيرة في كل من RNN وCNN (بعد تدريبهما).
- دمج (Concatenation، أو بديل أبسط: طبقة Attention صغيرة) → Dense نهائية → قرار موحّد (تأكيد نوع/شدة العيب بثقة مجمّعة).
- تقييم: مقارنة أداء/ثقة القرار الموحّد مقابل كل نموذج منفرد على نفس عيّنات الاختبار — هذا الجدول المقارن هو الدليل العلمي المطلوب لقيمة الـFusion.

### المرحلة 7 — LangGraph Agent
**تدفق الـGraph (بدون أي API بيانات خارجية؛ يوجد استدعاء LLM + أداة محلية واحدة فقط):**
```
START → Receive Input → هل يوجد نص؟ → Text Analyzer (RNN)
                      → هل توجد صورة؟ → Image Analyzer (CNN)
       → Information Extraction → Fusion → Context Builder
       → هل Requested_Action = استرجاع/استبدال؟ → [Tool] Local Policy Lookup (CSV)
       → LLM → Final Response
```

**مزوّد الـLLM:** بيئة الفريق محليًا مُعدّة مسبقًا مع أكثر من مزوّد عبر OpenRouter (وصول جاهز، لا حاجة لقرار إضافي) — المطلوب فقط الربط البرمجي بعقدة الـLLM الأخيرة بالـGraph.

**أداة Tool Use المحلية (جديدة):** `tools/policy_lookup.py` — تقرأ ملف `data/policies.csv` (schema: `product_category, return_window_days, warranty_summary`) وتُعيد النص المطابق لفئة المنتج. تُستدعى شرطيًا فقط عندما يشير `Requested_Action` (من مرحلة Information Extraction) إلى استرجاع/استبدال. هذا يُظهر نمط Tool Use بمعمارية الـAgent دون أي اعتماد على خدمة خارجية عبر الإنترنت.
- **System Prompt** (مسودة أولية جاهزة للاستخدام):
```
You are an AI Product Inspection Assistant. Analyze user text and image
analysis results. Extract important product and defect information.
Do not invent information. If confidence is low, clearly state the
uncertainty. Provide a concise and structured response.
```
- **Context Template** (مثال):
```
PRODUCT: <من Information Extraction>
IMAGE ANALYSIS: Defect: <type> | Location: <bbox/mask summary> | Confidence: <%>
TEXT ANALYSIS: Complaint: <problem_type> | Severity: <level> | Confidence: <%>
FUSION RESULT: <القرار الموحد بعد الدمج>
USER REQUEST: <Requested_Action>
```
- التعامل الصريح مع 3 حالات: نص فقط / صورة فقط / الاثنين معًا.

### المرحلة 8 — الواجهة + الاختبار الشامل
- واجهة تجريبية بسيطة (Streamlit أو Gradio) — رفع صورة + حقل نص + زر تحليل + عرض المخرجات المنظمة.
- اختبار End-to-End يغطي: نص فقط، صورة فقط، الاثنين، حالة "بدون عيب واضح" (Edge case).

### المرحلة 9 — التوثيق والتسليم
- ثلاث تقارير منفصلة (أو تقرير موحّد بأقسام واضحة) تربط كل عنصر تقني بمتطلب المادة المقابلة له.
- عرض تجريبي حي (Demo) يوضّح الـPipeline الكامل من الإدخال حتى التقرير النهائي.

---

## 7. ما هو مطلوب منّا (الفريق) قبل تسليم أي مرحلة للتنفيذ

- [ ] اختيار فئة MVTec النهائية للبدء (Bottle / Hazelnut / غيرها).
- [ ] تحميل MVTec AD يدويًا ووضعه في `data/raw/mvtec/` (الملف كبير، التحميل يحتاج قرار مسبق بمكان التخزين).
- [ ] الاتفاق على توزيع المهام بين الشخصين (نص/NLP مقابل صورة/Vision، أو تقسيم أفقي حسب المراحل).
- [ ] تثبيت البيئة (Python + المكتبات: TensorFlow/PyTorch, LangChain/LangGraph, scikit-learn, OpenCV, pandas).
- [x] مزوّد الـLLM: **محسوم** — بيئة محلية جاهزة مسبقًا مع أكثر من مزوّد عبر OpenRouter، يبقى فقط الربط البرمجي بعقدة الـAgent.
- [ ] إعداد ملف `data/policies.csv` (فئة المنتج + مدة الاسترجاع + ملخص الضمان) لدعم أداة Tool Use المحلية بالمرحلة 7.

## 8. ما يجب أن يكون جاهزًا لكي يبدأ الوكيل (AI Coding Agent) التنفيذ الكامل فورًا

لكي يبدأ وكيل ذكاء اصطناعي (مثل Claude Code) العمل مباشرة بدون توقف لطلب توضيحات، يجب توفر:

1. **هيكلية المجلدات** كما في القسم 6 / المرحلة 0 (يمكن للوكيل نفسه إنشاءها إذا طُلب منه ذلك صراحة كأول مهمة).
2. **بيانات MVTec فعليًا موجودة على القرص** بالمسار المتفق عليه (الوكيل لا يستطيع تحميلها من الإنترنت إن كان بيئة معزولة — يفضّل تنزيلها يدويًا أولًا).
3. **هذا الملف (PRD) + ملف ملخص النقاش** كسياق كامل يُرفق أول رسالة للوكيل.
4. ~~قرار نهائي بشأن مزوّد الـLLM~~ — **محسوم**: بيئة محلية جاهزة مسبقًا مع أكثر من مزوّد عبر OpenRouter، تحتاج فقط ربط برمجي.
5. **قرار بشأن فئة/فئات MVTec** المستخدمة (البند 7 أعلاه) — الوكيل يحتاج هذا صراحة قبل أي كود.
6. **ترتيب الأولوية**: هل يبدأ الوكيل بالمرحلة 1 (مولّد النصوص) كما هو مخطط، أم مرحلة أخرى؟
7. ملف `data/policies.csv` مُجهّز مسبقًا (بند 7 أعلاه) لأداة الـTool Use المحلية بالمرحلة 7.

**بمجرد توفر البنود 1-3 و5-7، يمكن للوكيل تنفيذ المراحل 1 إلى 9 بالتسلسل دون الحاجة لعودة متكررة للفريق. النقطة الوحيدة المتبقية كافتراض غير مؤكد صراحة هي بنية RNN الدقيقة (Shared Encoder) المذكورة بالقسم 5 من ملف ملخص النقاش — الوكيل يعتمدها كافتراض افتراضي ما لم يُطلب تغييرها.**

---

## 9. معايير القبول العامة للمشروع (Definition of Done)

- Dataset مدمجة موثّقة وقابلة لإعادة التوليد (Reproducible) عبر Script واحد.
- نموذج RNN ونموذج CNN مدرَّبان ومحفوظان مع تقارير تقييم واضحة (أرقام + Confusion Matrix / IoU-Dice).
- دليل كمّي (جدول مقارن) يثبت أن الـFusion يحسّن النتيجة مقارنة بكل نموذج منفرد.
- Agent يعمل End-to-End على الحالات الثلاث (نص/صورة/الاثنين) وينتج تقريرًا منظّمًا بدون اختلاق معلومات.
- توثيق يربط بوضوح كل مخرج تقني بمتطلب كل مادة من المواد الثلاث.
