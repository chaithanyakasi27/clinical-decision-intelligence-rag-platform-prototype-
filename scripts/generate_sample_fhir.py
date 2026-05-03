import json
import uuid
import random
from pathlib import Path
from datetime import date, timedelta

OUTPUT_DIR = Path("data/synthea_output/fhir")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIRST_NAMES_M = [
    "James", "Robert", "John", "Michael", "William",
    "David", "Aaron", "Carlos", "Brian", "Kevin",
]
FIRST_NAMES_F = [
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara",
    "Susan", "Maria", "Sandra", "Donna", "Carol",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones",
    "Garcia", "Miller", "Davis", "Abernathy", "Chen",
    "Christiansen", "Dominguez", "Anderson", "Wilson", "Moore",
    "Taylor", "Jackson", "Martin", "Lee", "Thompson",
]

CONDITION_SETS = [
    [
        ("E11.65", "Type 2 diabetes mellitus with hyperglycemia"),
        ("I50.9",  "Heart failure, unspecified"),
        ("N18.3",  "Chronic kidney disease, stage 3 (moderate)"),
        ("I10",    "Essential (primary) hypertension"),
    ],
    [
        ("E11.9",  "Type 2 diabetes mellitus without complications"),
        ("J44.1",  "COPD with acute exacerbation"),
        ("I10",    "Essential (primary) hypertension"),
        ("Z79.4",  "Long-term (current) use of insulin"),
    ],
    [
        ("N18.4",  "Chronic kidney disease, stage 4 (severe)"),
        ("E11.40", "Type 2 diabetes mellitus with diabetic neuropathy, unspecified"),
        ("I50.32", "Chronic diastolic (congestive) heart failure"),
        ("I10",    "Essential (primary) hypertension"),
    ],
    [
        ("J44.0",  "COPD with acute lower respiratory infection"),
        ("E11.9",  "Type 2 diabetes mellitus without complications"),
        ("N18.3",  "Chronic kidney disease, stage 3 (moderate)"),
        ("I10",    "Essential (primary) hypertension"),
    ],
    [
        ("E11.65", "Type 2 diabetes mellitus with hyperglycemia"),
        ("N18.5",  "Chronic kidney disease, stage 5"),
        ("I50.9",  "Heart failure, unspecified"),
        ("Z79.4",  "Long-term (current) use of insulin"),
    ],
]

MEDICATION_SETS = [
    [
        "Metformin 1000 MG Oral Tablet",
        "Lisinopril 10 MG Oral Tablet",
        "Furosemide 40 MG Oral Tablet",
    ],
    [
        "Insulin glargine 100 UNT/ML Injectable Solution",
        "Albuterol 0.83 MG/ML Inhalation Solution",
        "Amlodipine 5 MG Oral Tablet",
    ],
    [
        "Metformin 500 MG Oral Tablet",
        "Carvedilol 12.5 MG Oral Tablet",
        "Gabapentin 300 MG Oral Tablet",
    ],
    [
        "Atorvastatin 40 MG Oral Tablet",
        "Metoprolol Succinate 50 MG Oral Tablet",
        "Losartan 50 MG Oral Tablet",
    ],
    [
        "Insulin lispro 100 UNT/ML Injectable Solution",
        "Bumetanide 1 MG Oral Tablet",
        "Pantoprazole 40 MG Oral Tablet",
    ],
]


def random_dob():
    days = random.randint(55 * 365, 82 * 365)
    return (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")


def random_onset():
    days = random.randint(2 * 365, 15 * 365)
    return (date.today() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_bundle(index):
    gender     = random.choice(["male", "female"])
    first_name = random.choice(FIRST_NAMES_M if gender == "male" else FIRST_NAMES_F)
    last_name  = random.choice(LAST_NAMES)
    patient_id = str(uuid.uuid4())
    conditions  = CONDITION_SETS[index % len(CONDITION_SETS)]
    medications = MEDICATION_SETS[index % len(MEDICATION_SETS)]
    entries = []

    # Patient resource
    entries.append({"resource": {
        "resourceType": "Patient",
        "id":           patient_id,
        "name":         [{"given": [first_name], "family": last_name}],
        "birthDate":    random_dob(),
        "gender":       gender,
        "address":      [{"city": "Boston", "state": "Massachusetts", "postalCode": "02101"}],
    }})

    # Condition resources with ICD-10 codes
    for code, display in conditions:
        entries.append({"resource": {
            "resourceType": "Condition",
            "id":           str(uuid.uuid4()),
            "subject":      {"reference": f"Patient/{patient_id}"},
            "code": {
                "coding": [{
                    "system":  "http://hl7.org/fhir/sid/icd-10",
                    "code":    code,
                    "display": display,
                }],
                "text": display,
            },
            "clinicalStatus": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code":   "active",
                }]
            },
            "onsetDateTime": random_onset(),
            "recordedDate":  date.today().strftime("%Y-%m-%d"),
        }})

    # MedicationRequest resources
    for med in medications:
        entries.append({"resource": {
            "resourceType": "MedicationRequest",
            "id":           str(uuid.uuid4()),
            "status":       "active",
            "intent":       "order",
            "subject":      {"reference": f"Patient/{patient_id}"},
            "medicationCodeableConcept": {"text": med},
            "authoredOn":   date.today().strftime("%Y-%m-%d"),
        }})

    # Encounter resource
    entries.append({"resource": {
        "resourceType": "Encounter",
        "id":           str(uuid.uuid4()),
        "status":       "finished",
        "subject":      {"reference": f"Patient/{patient_id}"},
        "type":         [{"coding": [{"display": "Office Visit"}]}],
        "period": {
            "start": date.today().strftime("%Y-%m-%dT09:00:00Z"),
            "end":   date.today().strftime("%Y-%m-%dT09:30:00Z"),
        },
    }})

    # HbA1c Observation
    hba1c = round(random.uniform(7.2, 11.4), 1)
    entries.append({"resource": {
        "resourceType": "Observation",
        "id":           str(uuid.uuid4()),
        "status":       "final",
        "subject":      {"reference": f"Patient/{patient_id}"},
        "code": {
            "coding": [{
                "system":  "http://loinc.org",
                "code":    "4548-4",
                "display": "Hemoglobin A1c/Hemoglobin.total in Blood",
            }],
            "text": "Hemoglobin A1c",
        },
        "valueQuantity": {
            "value":  hba1c,
            "unit":   "%",
            "system": "http://unitsofmeasure.org",
        },
        "effectiveDateTime": date.today().strftime("%Y-%m-%dT09:15:00Z"),
    }})

    bundle = {
        "resourceType": "Bundle",
        "id":           str(uuid.uuid4()),
        "type":         "collection",
        "entry":        entries,
    }
    return bundle, f"{first_name}_{last_name}_{patient_id[:8]}"


def main():
    random.seed(42)

    print("=" * 55)
    print("  Synthetic FHIR R4 Generator (Synthea-compatible)")
    print("=" * 55)

    for i in range(50):
        bundle, fname = make_bundle(i)

        output_path = OUTPUT_DIR / f"{fname}.json"
        output_path.write_text(json.dumps(bundle, indent=2))

        patient = next(
            e["resource"] for e in bundle["entry"]
            if e["resource"]["resourceType"] == "Patient"
        )
        codes = [
            e["resource"]["code"]["coding"][0]["code"]
            for e in bundle["entry"]
            if e["resource"]["resourceType"] == "Condition"
        ]
        name = f"{patient['name'][0]['given'][0]} {patient['name'][0]['family']}"
        print(f"  + {name:<28} ICD-10: {', '.join(codes)}")

    files = list(OUTPUT_DIR.glob("*.json"))
    print()
    print(f"  Generated : {len(files)} FHIR bundles")
    print(f"  Location  : {OUTPUT_DIR.resolve()}")
    print("=" * 55)
    print("  Next      : python scripts/generate_clinical_pdfs.py")
    print("=" * 55)


if __name__ == "__main__":
    main()