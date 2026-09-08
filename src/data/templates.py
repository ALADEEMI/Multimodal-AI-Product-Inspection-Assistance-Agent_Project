"""Text Templates for Synthetic Complaint Generation.

Provides bilingual (Arabic/English) templates for generating synthetic complaint text
paired with MVTec defect images. Follows ADR Phase 1 specifications.
"""

from typing import Dict, List

# Arabic complaint templates for defects
ARABIC_TEMPLATES = {
    "broken_large": [
        "الزجاجة مكسورة بشكل كبير ولا يمكن استخدامها",
        "يوجد كسر واضح وكبير في المنتج، أريد استبدال",
        "المنتج وصل تالف، يوجد شرخ كبير في الزجاجة",
        "الزجاجة فيها كسر خطير ومش صالحة للاستعمال",
        "ارغب بإرجاع المنتج لان فيه كسر كبير جداً",
    ],
    "broken_small": [
        "في شرخ صغير في الزجاجة بس مزعج",
        "يوجد كسر بسيط لكنه واضح في المنتج",
        "الزجاجة فيها خدش او كسر صغير، ممكن استبدال؟",
        "لاحظت وجود شق صغير في الزجاجة عند الاستلام",
        "المنتج به عيب بسيط، كسر صغير بالزجاجة",
    ],
    "contamination": [
        "في اوساخ او تلوث داخل الزجاجة",
        "المنتج غير نظيف، يوجد شوائب واضحة",
        "الزجاجة ملوثة ولا تبدو نظيفة من الداخل",
        "لاحظت وجود اتربة او مواد غريبة بالمنتج",
        "جودة التنظيف سيئة، في تلوث واضح بالزجاجة",
    ],
    "good": [
        "المنتج وصل بحالة ممتازة، شكراً",
        "الزجاجة سليمة وجودتها جيدة",
        "كل شيء تمام، لا يوجد مشاكل",
        "المنتج مطابق للوصف وبحالة ممتازة",
    ],
}

# English complaint templates for defects
ENGLISH_TEMPLATES = {
    "broken_large": [
        "The bottle is severely broken and unusable",
        "There is a major crack in the product, I want a replacement",
        "Product arrived damaged with large breakage",
        "Bottle has serious damage, not suitable for use",
        "I need to return this, there's a huge crack in it",
    ],
    "broken_small": [
        "There's a small crack in the bottle but its annoying",
        "Minor damage visible on the product",
        "Bottle has a small chip or crack, can I get replacement?",
        "Noticed a small fracture upon receiving the bottle",
        "Product has minor defect, small crack on glass",
    ],
    "contamination": [
        "There is dirt or contamination inside the bottle",
        "Product is not clean, visible impurities present",
        "Bottle looks contaminated from inside",
        "I noticed dust or foreign material in the product",
        "Poor quality cleaning, obvious contamination",
    ],
    "good": [
        "Product arrived in excellent condition, thank you",
        "Bottle is intact and quality is good",
        "Everything is fine, no issues",
        "Product matches description and is in perfect condition",
    ],
}


def get_template(defect_type: str, language: str = "mixed") -> str:
    """Get a random template for the given defect type.

    Args:
        defect_type: One of 'good', 'broken_large', 'broken_small', 'contamination'
        language: 'arabic', 'english', or 'mixed' (default)

    Returns:
        Template string
    """
    import random

    if language == "mixed":
        # 50/50 chance for Arabic or English
        language = random.choice(["arabic", "english"])

    if language == "arabic":
        templates = ARABIC_TEMPLATES.get(defect_type, ARABIC_TEMPLATES["good"])
    else:
        templates = ENGLISH_TEMPLATES.get(defect_type, ENGLISH_TEMPLATES["good"])

    return random.choice(templates)


def get_all_templates() -> Dict[str, Dict[str, List[str]]]:
    """Return all templates for documentation purposes."""
    return {
        "arabic": ARABIC_TEMPLATES,
        "english": ENGLISH_TEMPLATES,
    }
