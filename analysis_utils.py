import re

def normalize_arabic(text):
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'[ى]', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[ًٌٍَُِّْ]', '', text)
    return text.lower()

AR_EMERGENCY = [
    "صعوبة التنفس",
    "ألم في الصدر",
    "فقدان الوعي",
    "تشنج",
    "تورم الوجه"
]

def arabic_emergency_detect(text):
    text = normalize_arabic(text)
    return any(k in text for k in AR_EMERGENCY)
