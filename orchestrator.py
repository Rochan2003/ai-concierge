import json
import os
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List


os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY_HERE"


# 1. DEFINE THE ENFORCED OUTPUT SCHEMA

class ActionItem(BaseModel):
    task: str = Field(description="A short, actionable title for the patient.")
    status: str = Field(description="Must be 'pending', 'in-progress', or 'completed'.")
    priority: str = Field(description="Must be 'high', 'medium', or 'low'.")
    clinical_reason: str = Field(
        description="A brief, patient-friendly explanation of why this is needed based on their FHIR data.")


class CarePlanUpdate(BaseModel):
    current_phase: str = Field(description="The current phase of treatment, e.g., 'Baseline Testing'.")
    missing_data: List[str] = Field(description="Any lab tests or data marked as 'registered' but lacking values.")
    action_items: List[ActionItem] = Field(description="The exact checklist to render in the UI.")



# 2. ORCHESTRATOR LOGIC

class PatientOrchestrator:
    def __init__(self):
        # Configure the Gemini API
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        genai.configure(api_key=api_key)

        # Initialize the model with the structured output schema
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",  # Using the fast model for UI interactions
            system_instruction="You are a clinical care coordinator AI. Analyze the provided FHIR JSON data. Identify what is complete and what is missing, and generate a structured CarePlan checklist for the patient."
        )

    def _load_json(self, filepath: str) -> dict:
        """Helper to load local FHIR mocks."""
        with open(filepath, 'r') as f:
            return json.load(f)

    def evaluate_patient_state(self, data_dir: str = "data") -> str:
        """Reads the IRIS Vault and determines the next steps."""

        # 1. Load the FHIR context
        patient = self._load_json(os.path.join(data_dir, "patient.json"))
        obs_amh = self._load_json(os.path.join(data_dir, "observation_amh.json"))
        obs_fsh = self._load_json(os.path.join(data_dir, "observation_fsh.json"))

        # 2. Construct the prompt with the injected data
        prompt = f"""
        Analyze the following patient data and generate the next steps for their CarePlan.

        Patient Profile:
        {json.dumps(patient, indent=2)}

        AMH Lab Result:
        {json.dumps(obs_amh, indent=2)}

        FSH Lab Result:
        {json.dumps(obs_fsh, indent=2)}

        Task:
        1. Note that the AMH result is final.
        2. Note that the FSH result is only 'registered' (pending completion).
        3. Create an action item for the patient to schedule their FSH bloodwork.
        4. Create an action item for an initial consultation to review the AMH results.
        """

        # 3. Call Gemini, strictly enforcing the Pydantic schema
        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=CarePlanUpdate,
                temperature=0.2  # Low temperature for deterministic, reliable output
            )
        )

        return response.text



# 3. EXECUTION FOR TESTING

if __name__ == "__main__":
    print("Initializing Orchestrator...")
    orchestrator = PatientOrchestrator()

    print("Evaluating FHIR Data and generating CarePlan update...")
    careplan_json = orchestrator.evaluate_patient_state()

    print("\n--- LLM JSON Output ---")
    print(careplan_json)

    # Optional: Save this output back to careplan_empty.json to complete the loop
    with open("data/careplan_empty.json", "w") as f:
        f.write(careplan_json)