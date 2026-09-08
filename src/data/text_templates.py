"""
Text Templates for Synthetic Complaint Generation.
Phase 1: Paired Dataset - Arabic and English templates with variations.
"""

from typing import Dict, List

# English Templates for each defect type
TEMPLATES_EN = {
    "good": [
        "The product looks fine and meets my expectations",
        "Everything seems to be in good condition",
        "No issues with the item, it arrived as described",
        "The quality is acceptable and no damage observed",
    ],
    "broken_large": [
        "The bottle is severely broken and completely unusable",
        "There is a large crack making this product defective",
        "Major damage observed, the item is shattered",
        "The product arrived with significant breakage",
        "This bottle has a huge break and cannot be used",
    ],
    "broken_small": [
        "I noticed a small crack on the bottle",
        "There is minor damage but it affects usability",
        "A small chip is visible on the product",
        "The item has a slight break that concerns me",
        "Minor breakage detected on arrival",
    ],
    "contamination": [
        "The bottle has visible contamination inside",
        "There are dirty marks and impurities on the product",
        "I found contamination that makes this unacceptable",
        "The item appears to have quality issues with stains",
        "Contamination is clearly visible and unacceptable",
    ],
}

# Arabic Templates for each defect type
TEMPLATES_AR = {
    "good": [
        "المنتج يبدو جيداً ويلبي توقعاتي",
        "كل شيء في حالة جيدة",
        "لا توجد مشاكل مع المنتج، وصل كما وُصف",
        "الجودة مقبولة ولا يوجد ضرر ملحوظ",
    ],
    "broken_large": [
        "الزجاجة مكسورة بشكل كبير وغير قابلة للاستخدام",
        "يوجد شرخ كبير يجعل المنتج معيباً",
        "ضرر كبير، المنتج محطم",
        "وصل المنتج مع كسر كبير",
        "الزجاجة بها كسر ضخم ولا يمكن استخدامها",
    ],
    "broken_small": [
        "لاحظت شرخاً صغيراً على الزجاجة",
        "يوجد ضرر بسيط لكنه يؤثر على الاستخدام",
        "توجد شقوق صغيرة مرئية على المنتج",
        "المنتج به كسر طفيف يقلقني",
        "كسر بسيط تم اكتشافه عند الوصول",
    ],
    "contamination": [
        "الزجاجة بها تلوث واضح من الداخل",
        "توجد علامات اتساخ وشوائب على المنتج",
        "وجدت تلوثاً يجعل هذا غير مقبول",
        "المنتج يبدو أن به مشاكل جودة مع بقع",
        "التلوث واضح تماماً وغير مقبول",
    ],
}

# Synonym variations for key words (English)
SYNONYMS_EN = {
    "bottle": ["bottle", "container", "product", "item"],
    "broken": ["broken", "cracked", "damaged", "shattered"],
    "contamination": ["contamination", "dirt", "impurity", "stain", "marks"],
    "good": ["good", "fine", "acceptable", "satisfactory"],
}

# Action keywords for Requested_Action extraction
ACTION_KEYWORDS_EN = {
    "return": ["return", "refund", "send back", "give back"],
    "replacement": ["replace", "exchange", "swap", "new one"],
    "repair": ["repair", "fix", "restore"],
}

ACTION_KEYWORDS_AR = {
    "return": ["إرجاع", "استرجاع", "إعادة"],
    "replacement": ["استبدال", "تبديل", "بديل"],
    "repair": ["إصلاح", "تصليح"],
}


def get_all_templates() -> Dict[str, List[str]]:
    """
    Combine English and Arabic templates.

    Returns:
        Dictionary mapping defect types to combined template lists
    """
    combined = {}
    for defect_type in TEMPLATES_EN.keys():
        combined[defect_type] = TEMPLATES_EN[defect_type] + TEMPLATES_AR[defect_type]
    return combined
