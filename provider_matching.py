import json
import os
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List



# 1. DEFINE THE OUTPUT SCHEMA

class ProviderMatch(BaseModel):
    provider_name: str
    specialty: str
    match_score: int = Field(description="Score from 0 to 100 based on fit.")
    reason: str = Field(
        description="Why this provider is a good match based on the patient's specific diagnosis and location.")
    next_step: str = Field(description="Actionable next step, e.g., 'Call clinic to verify insurance'.")


class MatchingResponse(BaseModel):
    matches: List[ProviderMatch]



# 2. THE MATCHING AGENT

class ProviderMatchingAgent:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction="You are a medical provider matching agent. Rank the provided doctors for the patient based on geographic proximity, clinical diagnosis fit, and care stage. Output a ranked list."
        )

    def _load_json(self, filepath: str) -> dict:
        with open(filepath, 'r') as f:
            return json.load(f)

    def find_best_matches(self, data_dir: str = "data") -> str:
        # Load Patient Data (Jane Doe, Boston, DOR diagnosis)
        patient = self._load_json(os.path.join(data_dir, "patient.json"))
        # Load Provider Network
        providers = self._load_json(os.path.join(data_dir, "providers.json"))

        prompt = f"""
        Patient Profile:
        {json.dumps(patient, indent=2)}

        Available Providers:
        {json.dumps(providers, indent=2)}

        Task: 
        Rank these providers for this specific patient. 
        - Give a high score to providers in the same city or telehealth.
        - Give a high score to providers whose focus_areas match the patient's clinicalDiagnosis.
        - Include both a medical doctor and an emotional support option if applicable.
        """

        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=MatchingResponse,
                temperature=0.1
            )
        )
        return response.text