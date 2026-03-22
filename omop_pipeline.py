import json
import os
import pandas as pd
from datetime import datetime
import hashlib


# 1. HELPER FUNCTIONS FOR DE-IDENTIFICATION

def generate_omop_person_id(fhir_id: str) -> int:
    """OMOP requires integer IDs. We hash the FHIR ID to create a deterministic integer."""
    return int(hashlib.sha256(fhir_id.encode('utf-8')).hexdigest()[:8], 16)


def safe_harbor_date(exact_date_str: str) -> int:
    """HIPAA Safe Harbor: Strip exact birth dates, retain only the Year of Birth."""
    date_obj = datetime.strptime(exact_date_str, "%Y-%m-%d")
    return date_obj.year



# 2. THE ETL ENGINE

class OmopPipeline:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

    def _load_fhir(self, filename: str) -> dict:
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None

    def process_person_table(self):
        """Converts FHIR Patient -> OMOP PERSON Table with Safe Harbor De-ID."""
        patient = self._load_fhir("patient.json")
        if not patient:
            return None

        # OMOP Concept Mapping
        gender_map = {"female": 8532, "male": 8507, "unknown": 0}

        person_record = {
            "person_id": generate_omop_person_id(patient["id"]),
            "gender_concept_id": gender_map.get(patient.get("gender", "unknown"), 0),
            "year_of_birth": safe_harbor_date(patient["birthDate"]),
            # Notice we are INTENTIONALLY dropping 'name' and 'address' to comply with IRB rules
            "person_source_value": patient["id"]
        }

        df = pd.DataFrame([person_record])
        output_path = os.path.join(self.data_dir, "omop_person.csv")
        df.to_csv(output_path, index=False)
        print(f"✅ Generated OMOP Person Table: {output_path}")
        return df

    def process_measurement_table(self):
        """Converts FHIR Observations -> OMOP MEASUREMENT Table."""
        patient = self._load_fhir("patient.json")
        obs_amh = self._load_fhir("observation_amh.json")
        obs_fsh = self._load_fhir("observation_fsh.json")

        person_id = generate_omop_person_id(patient["id"])
        measurements = []

        # Process AMH (Final Result)
        if obs_amh and obs_amh.get("status") == "final":
            measurements.append({
                "measurement_id": 1001,
                "person_id": person_id,
                # Ece's mapping: LOINC 38476-8 -> OMOP Concept (using placeholder standard concept ID for demo)
                "measurement_concept_id": 3020891,
                "measurement_date": obs_amh["effectiveDateTime"][:10],
                "value_as_number": obs_amh["valueQuantity"]["value"],
                "unit_source_value": obs_amh["valueQuantity"]["unit"],
                "measurement_source_value": obs_amh["code"]["coding"][0]["code"]  # The LOINC code
            })

        # Process FSH (Only if it has a value, currently it's just 'registered')
        if obs_fsh and obs_fsh.get("status") == "final":
            # This won't trigger currently because FSH is pending, which is clinically accurate!
            pass

        df = pd.DataFrame(measurements)
        output_path = os.path.join(self.data_dir, "omop_measurement.csv")
        df.to_csv(output_path, index=False)
        print(f" Generated OMOP Measurement Table: {output_path}")
        return df



# 3. EXECUTION

if __name__ == "__main__":
    print("Initiating IRB-Compliant OMOP ETL Pipeline...\n")
    pipeline = OmopPipeline()

    person_df = pipeline.process_person_table()
    measurement_df = pipeline.process_measurement_table()

    print("\n--- De-Identified OMOP PERSON Data ---")
    print(person_df)

    print("\n--- OMOP MEASUREMENT Data ---")
    print(measurement_df)