"""CheckList-style behavioural suite: invariance tests (INV) on real validation messages + hand-written minimum-functionality
tests (MFT). INV: the prediction must not change under a perturbation that should be irrelevant. MFT: short, unambiguous messages
written by hand (English, MSA Arabic and Gulf-dialect Arabic) with a known label."""
import re

import numpy as np

MFT_TOPIC = {
    "account_access": {
        "en": ["I cannot log in to my account", "I forgot my password and need to reset it", "My account was locked after several failed attempts",
               "The verification code never arrives when I sign in", "I want to change the mobile number registered on my account",
               "My account is suspended and I can't access it"],
        "ar": ["لا أستطيع تسجيل الدخول إلى حسابي", "نسيت كلمة المرور وأريد إعادة تعيينها", "تم قفل حسابي بعد عدة محاولات فاشلة",
               "لم يصلني رمز التحقق لتسجيل الدخول", "أريد تغيير رقم الجوال المسجل في حسابي", "الحساب موقوف ولا أستطيع الدخول"],
        "gulf": ["ما أقدر أدخل على حسابي أبد", "نسيت الرقم السري وش أسوي", "حسابي مقفل من أمس"]},
    "appointments": {
        "en": ["I need to book an appointment at the passport office", "Please reschedule my appointment to next week",
               "I want to cancel my clinic appointment", "No appointment slots are available this month",
               "My appointment was confirmed for Sunday but nobody attended to me", "How can I get an earlier appointment date"],
        "ar": ["أريد حجز موعد في مكتب الجوازات", "أرجو تأجيل موعدي إلى الأسبوع القادم", "أريد إلغاء موعدي في المركز الصحي",
               "لا توجد مواعيد متاحة هذا الشهر", "حجزت موعدا يوم الأحد ولم يتم استقبالي", "كيف أحصل على موعد أقرب"],
        "gulf": ["أبي أحجز موعد في الجوازات", "ودي أأجل موعدي لين الأسبوع الجاي", "ما لقيت موعد فاضي"]},
    "billing": {
        "en": ["I was charged twice for the same service", "My electricity bill is much higher than usual", "I did not receive my invoice this month",
               "Please refund the duplicate payment", "The amount on my bill is wrong", "I paid the fee but it still shows unpaid"],
        "ar": ["تم خصم المبلغ مرتين عن نفس الخدمة", "فاتورة الكهرباء أعلى بكثير من المعتاد", "لم تصلني فاتورة هذا الشهر",
               "أرجو استرداد المبلغ المدفوع مرتين", "المبلغ في الفاتورة غير صحيح", "دفعت الرسوم ولا تزال تظهر غير مسددة"],
        "gulf": ["انخصم مني المبلغ مرتين", "الفاتورة غالية مرة هالشهر", "دفعت وما زال يطلع علي مبلغ"]},
    "digital_services": {
        "en": ["The mobile app crashes when I open the services page", "The online portal shows an error when I submit the form",
               "I cannot upload documents on the website", "The e-service page keeps loading and never finishes",
               "The app does not show my application status", "The website is not working on my phone"],
        "ar": ["التطبيق يتوقف عند فتح صفحة الخدمات", "البوابة الإلكترونية تعرض خطأ عند إرسال النموذج", "لا أستطيع رفع المستندات على الموقع",
               "صفحة الخدمة الإلكترونية تظل تحمل ولا تكتمل", "التطبيق لا يعرض حالة طلبي", "الموقع لا يعمل على جوالي"],
        "gulf": ["التطبيق يعلق كل ما فتحته", "الموقع ما يشتغل على جوالي", "ما أقدر أرفع الملفات في البوابة"]},
    "outage": {
        "en": ["There has been no electricity in our neighbourhood since morning", "The water supply has been cut off for two days",
               "The internet is down in our area", "Power went out again last night", "No water in the building since yesterday",
               "Our street has had a blackout for hours"],
        "ar": ["انقطعت الكهرباء عن حينا منذ الصباح", "المياه مقطوعة عنا منذ يومين", "الإنترنت منقطع في منطقتنا",
               "انقطع التيار الكهربائي مرة أخرى الليلة الماضية", "لا توجد مياه في المبنى منذ أمس", "الشارع بدون كهرباء منذ ساعات"],
        "gulf": ["الكهرباء قاطعة عندنا من الصبح", "ما في موية من أمس", "النت طايح في حينا"]},
    "permits": {
        "en": ["I applied for a building permit three weeks ago and have no update", "How do I renew my business licence",
               "My construction permit application was rejected", "I need a permit to renovate my house", "The licence renewal is taking too long",
               "Which documents are needed for a commercial permit"],
        "ar": ["قدمت على رخصة بناء قبل ثلاثة أسابيع ولا يوجد أي تحديث", "كيف أجدد رخصة نشاطي التجاري", "تم رفض طلب رخصة البناء الخاصة بي",
               "أحتاج إلى تصريح لترميم منزلي", "تجديد الرخصة يستغرق وقتا طويلا", "ما هي المستندات المطلوبة لتصريح تجاري"],
        "gulf": ["قدمت على رخصة بناء وما أحد رد علي", "أبي أجدد رخصة المحل", "طلب التصريح تأخر مرة"]},
    "transport": {
        "en": ["The bus did not arrive for forty minutes", "The metro train was delayed again this morning",
               "The bus stop has no shelter and buses are always late", "The route changed and nobody told us",
               "The train was overcrowded and very late", "The bus driver skipped my stop"],
        "ar": ["لم تصل الحافلة لمدة أربعين دقيقة", "تأخر قطار المترو مرة أخرى هذا الصباح", "محطة الحافلات بلا مظلة والحافلات دائما متأخرة",
               "تغير المسار ولم يخبرنا أحد", "القطار كان مزدحما جدا ومتأخرا", "سائق الحافلة تجاوز محطتي"],
        "gulf": ["الباص تأخر أربعين دقيقة", "المترو متأخر اليوم بعد", "السواق عدى محطتي"]},
    "waste": {
        "en": ["The garbage has not been collected for a week", "The waste bins in our street are overflowing", "The collection truck missed our street again",
               "There is rubbish piling up next to the park", "Please empty the containers near my house",
               "Household waste collection is irregular in our area"],
        "ar": ["لم يتم جمع النفايات منذ أسبوع", "حاويات النفايات في شارعنا ممتلئة وتفيض", "شاحنة جمع النفايات تجاهلت شارعنا مرة أخرى",
               "القمامة تتراكم بجانب الحديقة", "أرجو تفريغ الحاويات القريبة من منزلي", "جمع النفايات المنزلية غير منتظم في منطقتنا"],
        "gulf": ["الزبالة ما انرفعت من أسبوع", "الحاويات مليانة عندنا", "سيارة الزبالة ما مرت علينا"]},
}
MFT_SENT = {
    "positive": {"en": ["Thank you, the service was excellent", "Very happy with how quickly the issue was solved", "The staff were helpful and polite",
                        "Great experience, the new app is easy to use"],
                 "ar": ["شكرا لكم الخدمة ممتازة", "سعيد جدا بسرعة حل المشكلة", "الموظفون متعاونون ومحترمون", "تجربة رائعة والتطبيق سهل الاستخدام"]},
    "negative": {"en": ["This is unacceptable, the service is terrible", "I am very disappointed with the long delay", "Worst experience ever, nobody helped me",
                        "I am angry that nobody answered my calls"],
                 "ar": ["هذا غير مقبول والخدمة سيئة جدا", "أنا محبط جدا من التأخير الطويل", "أسوأ تجربة ولم يساعدني أحد", "أنا غاضب لأن أحدا لم يرد على اتصالاتي"]},
    "neutral": {"en": ["I would like to know the working hours of the service centre", "Please tell me which documents are required",
                       "Could you share the contact details of the office", "I am asking about the status of my request"],
                "ar": ["أود معرفة ساعات عمل مركز الخدمة", "أرجو إفادتي بالمستندات المطلوبة", "هل يمكنكم مشاركة بيانات التواصل مع المكتب", "أستفسر عن حالة طلبي"]},
}
_PLACE = re.compile(r"\s-\s([^,،\-]+)[,،]\s")
_DIGITS = re.compile(r"\d+")
_AR_IND = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def mft_topic_cases():
    out = []
    for label, groups in MFT_TOPIC.items():
        for g, texts in groups.items():
            out += [{"text": t, "label": label, "group": g} for t in texts]
    return out


def mft_sent_cases():
    return [{"text": t, "label": l, "group": g} for l, groups in MFT_SENT.items() for g, texts in groups.items() for t in texts]


def run_mft(predict, cases):
    pred = predict([c["text"] for c in cases])
    rows = [dict(c, pred=p, ok=(p == c["label"])) for c, p in zip(cases, pred)]
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r["ok"])
    return {"pass_rate": float(np.mean([r["ok"] for r in rows])), "n": len(rows),
            "by_group": {g: {"pass_rate": float(np.mean(v)), "n": len(v)} for g, v in by_group.items()}, "failures": [r for r in rows if not r["ok"]]}


def _places(texts):
    return sorted({m.group(1).strip() for t in texts for m in [_PLACE.search(t)] if m})


def perturbations(texts, seed=42):
    """-> dict name -> (applicable_original_texts, perturbed_texts)."""
    rng = np.random.RandomState(seed)
    places = _places(texts)
    out = {}

    def add(name, fn, cond=lambda t: True):
        o = [t for t in texts if cond(t)]
        p = [fn(t) for t in o]
        keep = [(a, b) for a, b in zip(o, p) if a != b]
        out[name] = ([a for a, _ in keep], [b for _, b in keep])

    _ar = lambda x: bool(re.search(r"[\u0600-\u06FF]", x))  # noqa: E731

    def _swap(m):
        cur = m.group(1).strip()
        pool = [p for p in places if p != cur and _ar(p) == _ar(cur)]
        return " - " + str(rng.choice(pool)) + "، " if pool else m.group(0)

    add("swap_place_name", lambda t: _PLACE.sub(_swap, t, count=1), lambda t: bool(_PLACE.search(t)))
    add("change_numbers", lambda t: _DIGITS.sub(lambda m: str(rng.randint(2, 59)) if len(m.group()) <= 2 else m.group(), t, count=1),
        lambda t: bool(_DIGITS.search(t)))
    add("arabic_indic_digits", lambda t: t.translate(_AR_IND), lambda t: bool(_DIGITS.search(t)))
    add("trailing_period", lambda t: t.rstrip(". ") + ".")
    add("extra_whitespace", lambda t: "  ".join(t.split(" ")[:3]) + " " + " ".join(t.split(" ")[3:]))
    add("english_uppercase", lambda t: t.upper(), lambda t: not re.search(r"[؀-ۿ]", t))
    add("english_lowercase", lambda t: t.lower(), lambda t: not re.search(r"[؀-ۿ]", t))
    add("arabic_tatweel", lambda t: re.sub(r"([ء-ي])(?=[ء-ي]{2})", lambda m: m.group(1) + "ـ", t, count=1),
        lambda t: bool(re.search(r"[؀-ۿ]", t)))
    add("append_phone_number", lambda t: t + " 0551234567")
    add("append_email", lambda t: t + " contact a.b@example.com")
    add("append_national_id", lambda t: t + " 1012345678")
    return out


def run_invariance(predict, texts, n_max=300, seed=42):
    rng = np.random.RandomState(seed)
    texts = [texts[i] for i in rng.permutation(len(texts))[:max(n_max * 3, n_max)]]
    res, fails = {}, {}
    for name, (orig, pert) in perturbations(texts, seed).items():
        orig, pert = orig[:n_max], pert[:n_max]
        if not orig:
            continue
        a, b = predict(orig), predict(pert)
        same = [x == y for x, y in zip(a, b)]
        res[name] = {"invariance": float(np.mean(same)), "n": len(same)}
        fails[name] = [{"original": o, "perturbed": p, "pred_orig": x, "pred_pert": y} for o, p, x, y, s in zip(orig, pert, a, b, same) if not s][:8]
    res["_mean"] = float(np.mean([v["invariance"] for k, v in res.items()]))
    res["_min"] = float(min(v["invariance"] for k, v in res.items() if not k.startswith("_")))
    return res, fails
