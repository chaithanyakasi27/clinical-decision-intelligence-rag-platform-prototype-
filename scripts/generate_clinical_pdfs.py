
# ============================================================
# Clinical PDF Generator — Rich Multi-Section Notes
#
# Problem with previous version:
#   PDFs were only 350-400 characters — too small for sections.
#   pdf_parser detected 0 sections in all 20 PDFs.
#   Chunker fell back to full-text mode — no section awareness.
#
# This version generates:
#   - 3000-5000 characters per PDF
#   - 7 clinical sections with proper headers
#   - Realistic clinical narratives per condition
#   - ICD-10 codes in Assessment and Plan
#   - Multiple chunks per document after ingestion
# ============================================================

import json
import sys
import datetime
import random
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
FHIR_DIR   = Path("data/synthea_output/fhir")
OUTPUT_DIR  = Path("data/clinical_notes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── HCC reference data ────────────────────────────────────────
HCC_MAP = {
    "E11.9":  ("19",  "Diabetes without Complication",            0.104),
    "E11.65": ("19",  "Diabetes with Hyperglycemia",              0.104),
    "E11.40": ("18",  "Diabetes with Diabetic Neuropathy",        0.302),
    "I50.9":  ("85",  "Congestive Heart Failure",                 0.323),
    "I50.32": ("86",  "Heart Failure - Reduced EF",               0.368),
    "N18.3":  ("136", "Chronic Kidney Disease Stage 3",           0.289),
    "N18.4":  ("137", "Chronic Kidney Disease Stage 4",           0.421),
    "N18.5":  ("138", "Chronic Kidney Disease Stage 5",           0.519),
    "J44.1":  ("111", "COPD with Acute Exacerbation",             0.335),
    "J44.0":  ("111", "COPD with Acute Resp Infection",           0.335),
    "I10":    (None,  "Essential Hypertension",                   0.000),
    "Z79.4":  (None,  "Long-term Insulin Use",                    0.000),
}

# ── Detailed clinical narratives per condition ─────────────────
# These make the HPI and Assessment sections rich and realistic

CONDITION_NARRATIVES = {
    "E11.65": {
        "hpi": (
            "Patient reports ongoing difficulty with glycemic control despite "
            "compliance with prescribed Metformin therapy. Reports occasional "
            "episodes of hyperglycemia with blood glucose readings ranging from "
            "180-280 mg/dL at home. Denies hypoglycemic episodes. HbA1c at last "
            "check was 9.2%, indicating poorly controlled Type 2 diabetes mellitus "
            "with hyperglycemia. Patient reports increased thirst and urinary "
            "frequency consistent with hyperglycemic state."
        ),
        "plan": (
            "Type 2 diabetes mellitus with hyperglycemia (ICD-10: E11.65, HCC 19, RAF 0.104). "
            "HbA1c ordered — target less than 7.0 percent per ADA guidelines. "
            "Foot examination performed — no ulcers, calluses, or neuropathic changes noted. "
            "Metformin dose optimized. Referral to endocrinology placed for insulin initiation "
            "consideration. Diabetes self-management education reinforced. "
            "Ophthalmology referral placed for annual diabetic retinopathy screening. "
            "Patient counseled on dietary modifications and carbohydrate counting."
        ),
    },
    "E11.9": {
        "hpi": (
            "Patient presents for routine management of Type 2 diabetes mellitus "
            "without complications. Reports compliance with Metformin 1000mg twice "
            "daily. Home blood glucose monitoring shows fasting readings between "
            "110-140 mg/dL. Last HbA1c was 7.1%, within acceptable range. "
            "Patient denies polyuria, polydipsia, or blurred vision. "
            "No hypoglycemic episodes reported since last visit."
        ),
        "plan": (
            "Type 2 diabetes mellitus without complications (ICD-10: E11.9, HCC 19, RAF 0.104). "
            "Continue Metformin 1000mg twice daily. HbA1c ordered — target less than 7.0%. "
            "Annual foot exam performed — neurological and vascular exam normal. "
            "Continue annual ophthalmology screening. Dietary compliance reinforced. "
            "Patient instructed to monitor fasting glucose daily and log results."
        ),
    },
    "E11.40": {
        "hpi": (
            "Patient with known Type 2 diabetes mellitus now presenting with symptoms "
            "consistent with diabetic neuropathy. Reports bilateral lower extremity "
            "tingling and burning sensation, worse at night, rated 5/10 severity. "
            "Symptoms have been progressive over the past 8 months. "
            "Monofilament testing reveals diminished sensation in bilateral feet. "
            "HbA1c elevated at 10.1%, indicating poor glycemic control contributing "
            "to neuropathic progression."
        ),
        "plan": (
            "Type 2 diabetes mellitus with diabetic neuropathy (ICD-10: E11.40, HCC 18, RAF 0.302). "
            "Gabapentin initiated at 300mg nightly for neuropathic pain management. "
            "HbA1c ordered — aggressive glycemic control target less than 7.0% to slow progression. "
            "Podiatry referral placed for neuropathy management. "
            "Patient educated on daily foot inspection. Proper footwear counseled. "
            "Physical therapy referral placed for balance and fall prevention."
        ),
    },
    "I50.9": {
        "hpi": (
            "Patient with known congestive heart failure presenting for routine follow-up. "
            "Reports mild exertional dyspnea with activities of daily living, NYHA Class II. "
            "Denies orthopnea or paroxysmal nocturnal dyspnea at rest. "
            "Lower extremity edema present, rated 1+ bilateral pitting edema at ankles. "
            "Patient reports compliance with daily weight monitoring — no weight gain "
            "greater than 3 pounds in 24 hours reported. Last echocardiogram showed "
            "ejection fraction of 45%, mildly reduced."
        ),
        "plan": (
            "Congestive heart failure, unspecified (ICD-10: I50.9, HCC 85, RAF 0.323). "
            "Daily weight monitoring — patient instructed to call if weight gain exceeds 3 lbs/day. "
            "Dietary sodium restriction less than 2g per day reinforced. "
            "Furosemide dose reviewed and optimized for volume management. "
            "Echocardiogram ordered for interval assessment of ejection fraction. "
            "Cardiology follow-up scheduled in 4 weeks. "
            "Patient counseled on activity restriction and fluid management."
        ),
    },
    "I50.32": {
        "hpi": (
            "Patient with chronic diastolic heart failure presenting with worsening "
            "exertional dyspnea over the past 2 weeks. Reports shortness of breath "
            "with minimal exertion, NYHA Class III. Bilateral lower extremity edema "
            "noted, 2+ pitting edema to mid-calf. Patient reports 4-pound weight gain "
            "over 3 days prior to visit. Last echocardiogram demonstrated reduced "
            "ejection fraction of 35% with diastolic dysfunction."
        ),
        "plan": (
            "Chronic diastolic congestive heart failure (ICD-10: I50.32, HCC 86, RAF 0.368). "
            "Furosemide dose increased for acute volume management. "
            "Daily weights — threshold for calling clinic set at 2 lbs/day. "
            "Sodium restriction less than 1.5g per day. Fluid restriction 1.5L daily. "
            "Cardiology urgent referral placed. Repeat echocardiogram ordered. "
            "Patient admitted for IV diuresis consideration if no improvement in 48 hours."
        ),
    },
    "N18.3": {
        "hpi": (
            "Patient with known Chronic Kidney Disease Stage 3 presenting for nephrology "
            "co-management visit. Most recent BMP shows creatinine 1.8 mg/dL with "
            "estimated GFR of 42 mL/min/1.73m2, consistent with Stage 3 CKD. "
            "Patient denies dysuria, hematuria, or flank pain. "
            "Blood pressure well-controlled on current regimen. "
            "Urine protein to creatinine ratio 0.3, indicating mild proteinuria."
        ),
        "plan": (
            "Chronic kidney disease, Stage 3 moderate (ICD-10: N18.3, HCC 136, RAF 0.289). "
            "BMP ordered — monitoring GFR trend and electrolyte status quarterly. "
            "Nephrology co-management continued. Blood pressure target less than 130/80 mmHg. "
            "ACE inhibitor continued for renoprotective effect. "
            "All medication doses reviewed and renally dosed appropriately. "
            "NSAIDs and nephrotoxic agents strictly avoided. "
            "Dietary protein restriction 0.8g/kg/day counseled. Renal dietitian referral placed."
        ),
    },
    "N18.4": {
        "hpi": (
            "Patient with progressive Chronic Kidney Disease now classified as Stage 4. "
            "Recent BMP reveals creatinine 2.9 mg/dL with eGFR 22 mL/min/1.73m2. "
            "Patient reports fatigue, mild nausea, and decreased appetite over past month. "
            "Hyperkalemia noted on recent labs with potassium 5.6 mEq/L. "
            "Patient referred to nephrology for renal replacement therapy planning. "
            "Discussing arteriovenous fistula creation in anticipation of hemodialysis need."
        ),
        "plan": (
            "Chronic kidney disease, Stage 4 severe (ICD-10: N18.4, HCC 137, RAF 0.421). "
            "Urgent nephrology follow-up for renal replacement therapy planning. "
            "Potassium restriction less than 2g per day — dietitian referral placed. "
            "Phosphate binder initiated. Erythropoietin stimulating agent considered for anemia. "
            "AV fistula surgery referral placed for hemodialysis access preparation. "
            "All nephrotoxic medications discontinued. Contrast agents contraindicated."
        ),
    },
    "N18.5": {
        "hpi": (
            "Patient with end-stage Chronic Kidney Disease Stage 5, currently on "
            "peritoneal dialysis three times weekly. eGFR less than 10 mL/min/1.73m2. "
            "Reports adequate dialysis tolerance with no peritonitis episodes this quarter. "
            "Ongoing anemia managed with erythropoietin stimulating agent. "
            "Patient evaluated for kidney transplant listing — currently on waitlist."
        ),
        "plan": (
            "Chronic kidney disease, Stage 5 (ICD-10: N18.5, HCC 138, RAF 0.519). "
            "Continue peritoneal dialysis per current schedule. "
            "Nephrology follow-up monthly for dialysis adequacy assessment. "
            "Renal transplant evaluation continued — patient on active waitlist. "
            "Erythropoietin dose adjusted per hemoglobin target 10-11.5 g/dL. "
            "Dietary restrictions reinforced — potassium, phosphate, fluid. "
            "Vascular access monitoring continued."
        ),
    },
    "J44.1": {
        "hpi": (
            "Patient with known COPD presenting with acute exacerbation. Reports "
            "increased dyspnea over the past 5 days, productive cough with yellow-green "
            "sputum, and decreased exercise tolerance. Oxygen saturation 91% on room air. "
            "Rescue albuterol use increased to 6-8 times daily from baseline 1-2 times. "
            "No fever. Chest X-ray shows hyperinflation consistent with COPD without "
            "acute pneumonic infiltrate."
        ),
        "plan": (
            "COPD with acute exacerbation (ICD-10: J44.1, HCC 111, RAF 0.335). "
            "Systemic corticosteroids initiated — prednisone 40mg daily for 5 days. "
            "Azithromycin 500mg daily for 5 days for bacterial exacerbation coverage. "
            "Albuterol nebulization every 4 hours while symptomatic. "
            "Oxygen supplementation to maintain saturation above 92%. "
            "Pulmonology follow-up in 2 weeks. Smoking cessation counseling provided. "
            "Influenza and pneumococcal vaccines reviewed — both current."
        ),
    },
    "J44.0": {
        "hpi": (
            "Patient with COPD presenting with acute lower respiratory tract infection. "
            "Reports 7-day history of productive cough with green sputum, subjective fever, "
            "and worsening shortness of breath. Temperature 38.2°C. Oxygen saturation 90% "
            "on room air. Pulmonary function testing last year showed FEV1/FVC ratio 0.62, "
            "consistent with moderate obstructive pattern."
        ),
        "plan": (
            "COPD with acute lower respiratory infection (ICD-10: J44.0, HCC 111, RAF 0.335). "
            "Antibiotic therapy initiated — amoxicillin-clavulanate 875mg twice daily for 7 days. "
            "Short course systemic corticosteroids initiated. Bronchodilator therapy optimized. "
            "Chest physiotherapy ordered. Supplemental oxygen 2L nasal cannula. "
            "Return precautions given — instructed to go to emergency room if dyspnea worsens."
        ),
    },
    "I10": {
        "hpi": (
            "Patient with well-controlled essential hypertension on current antihypertensive "
            "regimen. Home blood pressure log shows readings averaging 128/78 mmHg. "
            "Denies headache, visual changes, or chest pain. "
            "No orthostatic symptoms reported. Medication compliance confirmed."
        ),
        "plan": (
            "Essential primary hypertension (ICD-10: I10). "
            "Continue current antihypertensive regimen. Blood pressure target less than 130/80 mmHg. "
            "BMP ordered to monitor electrolytes and renal function on ACE inhibitor. "
            "Patient encouraged to continue low sodium diet and regular aerobic exercise. "
            "Home blood pressure monitoring continued with log review at next visit."
        ),
    },
    "Z79.4": {
        "hpi": (
            "Patient on long-term insulin therapy for diabetes management. "
            "Currently using insulin glargine 20 units at bedtime with insulin lispro "
            "sliding scale with meals. Reports occasional mild hypoglycemia with glucose "
            "readings in 60-70 mg/dL range, self-managed with oral glucose. "
            "No severe hypoglycemic episodes requiring assistance."
        ),
        "plan": (
            "Long-term current use of insulin (ICD-10: Z79.4). "
            "Insulin regimen reviewed and adjusted. Hypoglycemia threshold counseling provided. "
            "Glucagon emergency kit prescribed. Continuous glucose monitoring discussed. "
            "Patient instructed never to skip meals after insulin administration. "
            "Medical alert bracelet recommended."
        ),
    },
}

# Default narrative for any condition not in our map
DEFAULT_NARRATIVE = {
    "hpi": (
        "Patient presents for routine chronic disease management. "
        "Reports stable symptoms with current therapeutic regimen. "
        "No acute decompensation or new symptoms reported since last visit. "
        "Medication compliance confirmed. Vital signs stable."
    ),
    "plan": (
        "Continue current therapeutic regimen as prescribed. "
        "Patient education reinforced regarding disease self-management. "
        "Monitoring plan reviewed and labs ordered as clinically indicated. "
        "Patient instructed to seek care promptly if symptoms worsen or change."
    ),
}


def parse_fhir(path: Path) -> dict:
    """Extract patient data from FHIR bundle JSON file."""
    bundle  = json.loads(path.read_text())
    entries = bundle.get("entry", [])

    patient    = {}
    conditions = []
    meds       = []

    for entry in entries:
        r  = entry.get("resource", {})
        rt = r.get("resourceType", "")

        if rt == "Patient":
            names      = r.get("name", [{}])
            name       = names[0] if names else {}
            given      = name.get("given", ["Unknown"])
            patient    = {
                "id"        : r.get("id", "unknown")[:12],
                "first_name": given[0] if given else "Unknown",
                "last_name" : name.get("family", "Unknown"),
                "dob"       : r.get("birthDate", "Unknown"),
                "gender"    : r.get("gender", "unknown").title(),
            }

        elif rt == "Condition":
            codings = r.get("code", {}).get("coding", [])
            for coding in codings:
                code    = coding.get("code", "")
                display = coding.get("display", "Unknown condition")
                if code:
                    onset  = r.get("onsetDateTime", "")[:10]
                    status = (
                        r.get("clinicalStatus", {})
                         .get("coding", [{}])[0]
                         .get("code", "active")
                    )
                    conditions.append({
                        "code"   : code,
                        "display": display,
                        "onset"  : onset,
                        "status" : status.upper(),
                    })
                    break

        elif rt == "MedicationRequest":
            med_name = (
                r.get("medicationCodeableConcept", {}).get("text", "")
                or r.get("medicationCodeableConcept", {})
                    .get("coding", [{}])[0]
                    .get("display", "")
            )
            if med_name:
                meds.append(med_name)

    return {
        "patient"   : patient,
        "conditions": conditions,
        "medications": meds,
    }


def calc_age(dob: str) -> str:
    """Calculate age from date of birth string."""
    try:
        d = datetime.datetime.strptime(dob, "%Y-%m-%d").date()
        return str((datetime.date.today() - d).days // 365)
    except Exception:
        return "N/A"


def build_note_text(data: dict) -> str:
    """
    Build rich clinical note text with all 7 sections.

    This is the key fix — we write plain text with section headers
    that pdf_parser.py can detect (lowercase matching).
    Each section has substantial content so chunking produces
    multiple meaningful chunks per document.
    """
    pt         = data["patient"]
    conditions = data["conditions"]
    meds       = data["medications"]
    visit_date = datetime.date.today().strftime("%B %d, %Y")
    age        = calc_age(pt.get("dob", ""))
    gender     = pt.get("gender", "Unknown").lower()
    first_name = pt.get("first_name", "Patient")
    last_name  = pt.get("last_name", "")
    patient_id = pt.get("id", "unknown")

    # Build combined HPI from all conditions
    hpi_parts = []
    for cond in conditions:
        narrative = CONDITION_NARRATIVES.get(cond["code"], DEFAULT_NARRATIVE)
        hpi_parts.append(narrative["hpi"])

    # Build Assessment and Plan from all conditions
    plan_parts = []
    hcc_conditions = []
    total_raf = 0.0

    for cond in conditions:
        narrative = CONDITION_NARRATIVES.get(cond["code"], DEFAULT_NARRATIVE)
        plan_parts.append(narrative["plan"])

        # Collect HCC-mapped conditions for summary
        hcc_info = HCC_MAP.get(cond["code"])
        if hcc_info and hcc_info[2] > 0:
            hcc_conditions.append({
                "code"     : cond["code"],
                "display"  : cond["display"],
                "hcc_num"  : hcc_info[0],
                "hcc_label": hcc_info[1],
                "raf"      : hcc_info[2],
            })
            total_raf += hcc_info[2]

    # ── Build the full note as plain text ─────────────────────
    # Section headers must match CLINICAL_SECTIONS in pdf_parser.py
    # They are detected by lowercase matching of the line

    lines = []

    # Hospital header
    lines.append("MASSACHUSETTS GENERAL HOSPITAL")
    lines.append("55 Fruit Street, Boston MA 02114 | (617) 726-2000")
    lines.append("AMBULATORY PROGRESS NOTE")
    lines.append("")

    # Demographics block
    lines.append(f"Patient: {first_name} {last_name}")
    lines.append(f"Patient ID: {patient_id}")
    lines.append(f"Date of Birth: {pt.get('dob', 'Unknown')} (Age {age})")
    lines.append(f"Gender: {pt.get('gender', 'Unknown')}")
    lines.append(f"Visit Date: {visit_date}")
    lines.append(f"Provider: Dr. Sarah Chen, MD — Internal Medicine")
    lines.append("")

    # ── Section 1: Chief Complaint ────────────────────────────
    lines.append("CHIEF COMPLAINT")
    lines.append(
        f"{first_name} is a {age}-year-old {gender} presenting for routine "
        f"chronic disease management, medication reconciliation, and annual "
        f"wellness review of documented chronic conditions."
    )
    lines.append("")

    # ── Section 2: History of Present Illness ─────────────────
    lines.append("HISTORY OF PRESENT ILLNESS")
    lines.append(
        f"{first_name} is a {age}-year-old {gender} with a known history of "
        f"{', '.join(c['display'] for c in conditions[:3])}. "
        f"Patient presents today for ongoing management of all documented "
        f"chronic conditions. The following clinical details pertain to each "
        f"active diagnosis:"
    )
    lines.append("")
    for part in hpi_parts:
        lines.append(part)
        lines.append("")

    # ── Section 3: Past Medical History ───────────────────────
    lines.append("PAST MEDICAL HISTORY")
    for cond in conditions:
        onset_str = f" (onset {cond['onset']})" if cond["onset"] else ""
        lines.append(
            f"- {cond['display']}{onset_str} — "
            f"ICD-10: {cond['code']} — Status: {cond['status']}"
        )
    lines.append("")

    # ── Section 4: Current Medications ────────────────────────
    lines.append("CURRENT MEDICATIONS")
    if meds:
        for med in meds:
            lines.append(f"- {med}")
    else:
        lines.append("No active medications on file.")
    lines.append("")

    # ── Section 5: Review of Systems ──────────────────────────
    lines.append("REVIEW OF SYSTEMS")
    lines.append(
        "Constitutional: No fever, chills, or unintentional weight loss. "
        "Cardiovascular: No chest pain, palpitations, or lower extremity edema beyond baseline. "
        "Respiratory: No acute dyspnea at rest or productive cough beyond baseline. "
        "Gastrointestinal: No nausea, vomiting, or abdominal pain. "
        "Endocrine: Denies polydipsia, polyuria, or diaphoresis beyond baseline. "
        "Renal/GU: No dysuria, hematuria, or significant change in voiding pattern. "
        "Neurological: No syncope, dizziness, or new focal neurological deficits. "
        "Musculoskeletal: No new joint pain or significant reduction in mobility."
    )
    lines.append("")

    # ── Section 6: Assessment and Plan ────────────────────────
    # This is the most important section for HCC coding
    lines.append("ASSESSMENT AND PLAN")
    lines.append(
        "The following diagnoses are active, clinically evaluated, and "
        "documented at today's visit per CMS HCC V28 Risk Adjustment guidelines. "
        "All conditions listed below are supported by clinical documentation "
        "and meet medical necessity for coding:"
    )
    lines.append("")

    for i, (cond, plan) in enumerate(zip(conditions, plan_parts), 1):
        hcc_info = HCC_MAP.get(cond["code"])
        hcc_tag  = ""
        if hcc_info and hcc_info[0]:
            hcc_tag = f" | HCC {hcc_info[0]} | RAF {hcc_info[2]:.3f}"

        lines.append(
            f"{i}. {cond['display']} — "
            f"ICD-10: {cond['code']}{hcc_tag}"
        )
        lines.append(plan)
        lines.append("")

    # HCC Risk Summary
    if hcc_conditions:
        lines.append("HCC RISK ADJUSTMENT SUMMARY")
        lines.append("The following conditions map to CMS HCC categories:")
        for hcc in hcc_conditions:
            lines.append(
                f"- {hcc['code']}: {hcc['display']} "
                f"→ HCC {hcc['hcc_num']} ({hcc['hcc_label']}) "
                f"RAF Score: {hcc['raf']:.3f}"
            )
        lines.append(f"Total Risk Adjustment Factor (RAF): {total_raf:.3f}")
        lines.append("")

    # ── Section 7: Follow-up Plan ──────────────────────────────
    lines.append("FOLLOW-UP PLAN")
    lines.append(
        f"- Return to clinic in 3 months for chronic disease follow-up. "
        f"- Labs ordered: HbA1c, BMP, CBC, Lipid Panel, eGFR as clinically indicated. "
        f"- Patient instructed to seek emergency care for acute decompensation. "
        f"- All specialist referrals placed as documented in Assessment and Plan above. "
        f"- Patient verbalized understanding of discharge instructions."
    )
    lines.append("")

    # Signature block
    lines.append(
        f"Electronically signed by: Dr. Sarah Chen, MD | "
        f"Internal Medicine | NPI: 1234567890 | {visit_date}"
    )
    lines.append(
        "CONFIDENTIAL — Contains protected health information. "
        "Unauthorized disclosure prohibited by HIPAA (45 CFR 164)."
    )

    return "\n".join(lines)


def build_pdf(text: str, output_path: Path) -> None:
    """
    Write clinical note text to PDF using ReportLab.

    The text is pre-structured with section headers that
    pdf_parser.py will detect via lowercase matching.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize     = letter,
        leftMargin   = 0.75 * inch,
        rightMargin  = 0.75 * inch,
        topMargin    = 0.75 * inch,
        bottomMargin = 0.75 * inch,
    )

    styles = getSampleStyleSheet()

    # Section header style — bold, slightly larger
    section_style = ParagraphStyle(
        "Section",
        parent    = styles["Normal"],
        fontName  = "Helvetica-Bold",
        fontSize  = 10,
        spaceBefore = 8,
        spaceAfter  = 4,
    )

    # Body text style
    body_style = ParagraphStyle(
        "Body",
        parent    = styles["Normal"],
        fontName  = "Helvetica",
        fontSize  = 9,
        leading   = 13,
        spaceAfter= 3,
    )

    # Title style for hospital header
    title_style = ParagraphStyle(
        "Title",
        parent    = styles["Normal"],
        fontName  = "Helvetica-Bold",
        fontSize  = 12,
        alignment = 1,  # center
        spaceAfter= 6,
    )

    story = []

    # These are the section header names that pdf_parser detects
    SECTION_HEADERS = {
        "chief complaint",
        "history of present illness",
        "past medical history",
        "current medications",
        "review of systems",
        "assessment and plan",
        "hcc risk adjustment summary",
        "follow-up plan",
    }

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 4))
            continue

        # Check if this line is a section header
        if stripped.lower() in SECTION_HEADERS:
            story.append(Paragraph(stripped, section_style))
        elif stripped in (
            "MASSACHUSETTS GENERAL HOSPITAL",
            "AMBULATORY PROGRESS NOTE",
        ):
            story.append(Paragraph(stripped, title_style))
        else:
            # Escape any HTML special characters for ReportLab
            safe = (stripped
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))
            story.append(Paragraph(safe, body_style))

    doc.build(story)


def main():
    print("=" * 55)
    print("  Clinical PDF Generator — Rich Multi-Section Notes")
    print("=" * 55)

    fhir_files = list(FHIR_DIR.glob("*.json"))
    if not fhir_files:
        print(f"ERROR: No FHIR files found in {FHIR_DIR}")
        print("Run: python scripts/generate_sample_fhir.py first")
        sys.exit(1)

    print(f"\n  FHIR files   : {len(fhir_files)}")
    print(f"  Output dir   : {OUTPUT_DIR.resolve()}")
    print()

    generated = 0
    failed    = 0

    for fhir_file in sorted(fhir_files)[:20]:
        try:
            data       = parse_fhir(fhir_file)
            pt         = data["patient"]
            name       = f"{pt.get('first_name','')} {pt.get('last_name','')}"
            patient_id = pt.get("id", fhir_file.stem[:8])

            # Build rich clinical note text
            note_text   = build_note_text(data)
            output_path = OUTPUT_DIR / f"{patient_id}_progress_note.pdf"

            # Write to PDF
            build_pdf(note_text, output_path)

            char_count = len(note_text)
            codes      = [c["code"] for c in data["conditions"]]
            print(
                f"  + {name:<28} "
                f"{', '.join(codes):<26} "
                f"{char_count:5d} chars"
            )
            generated += 1

        except Exception as e:
            print(f"  ERROR {fhir_file.name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print(f"  Generated : {generated} PDFs")
    if failed:
        print(f"  Failed    : {failed} PDFs")

    total_kb = sum(
        f.stat().st_size for f in OUTPUT_DIR.glob("*.pdf")
    ) // 1024
    print(f"  Total size: {total_kb} KB")
    print("=" * 55)
    print("  Next: python scripts/run_ingestion.py --force")
    print("=" * 55)


if __name__ == "__main__":
    main()