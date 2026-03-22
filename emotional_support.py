import json
import os
import google.generativeai as genai
from pydantic import BaseModel, Field


# 1. ENFORCED SCHEMA FOR SAFETY & BOUNDARIES

class EmotionalIntervention(BaseModel):
    is_distressed: bool = Field(
        description="True if the user's message indicates anxiety, sadness, frustration, or fear.")
    distress_category: str = Field(
        description="Categorize the moment: 'lab_anxiety', 'financial_stress', 'cycle_delay', 'loss', or 'none'.")
    requires_escalation: bool = Field(
        description="CRITICAL: True if the user mentions self-harm, severe pain, or urgent medical symptoms. False otherwise.")
    empathetic_response: str = Field(
        description="A brief, supportive response. MUST NOT contain medical advice. Normalize the feeling.")
    recommended_resource: str = Field(
        description="Suggest a next step: e.g., 'Contact Clinic Nurse', 'Join Resolve Support Group', 'Speak to Financial Counselor', or 'None'.")



# 2. THE EMPATHY GATEKEEPER

class EmpathyGatekeeper:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=api_key)

        # We use a strict system instruction to enforce the Option 5 Rubric boundaries
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=(
                "You are an empathetic AI Concierge for a fertility clinic. "
                "YOUR BOUNDARIES: You are NOT a doctor or therapist. You cannot interpret lab results, give medical advice, or tell a patient to stop treatment. "
                "YOUR RESPONSE STRUCTURE MUST BE: "
                "1. Validate and normalize the emotion. "
                "2. Gently pivot away from medical conclusions. "
                "3. ALWAYS end your response by offering a concrete next step (e.g., 'Let's have the clinical team review this with you before making any decisions. Would you like me to ping the triage nurse, or would you prefer the number for our counseling team?'). "
                "Never leave the patient hanging without a human lifeline."
            )
        )

    def _check_consent(self, data_dir: str = "data") -> bool:
        """Reads the FHIR Consent resource to ensure we are allowed to use Affective AI."""
        try:
            with open(os.path.join(data_dir, "consent.json"), 'r') as f:
                consent = json.load(f)
                # Check if the policy rule explicitly allows affective AI support
                rule = consent.get("policyRule", {}).get("coding", [{}])[0].get("code")
                return rule == "allow-affective-ai-support" and consent.get("status") == "active"
        except Exception:
            return False

    def analyze_patient_message(self, user_message: str, patient_context: dict) -> dict:
        """
        Analyzes the chat message for emotional distress and returns a safe intervention.
        """
        # 1. Safety/Compliance Gate
        if not self._check_consent():
            return {
                "is_distressed": False,
                "error": "Patient has not consented to affective AI analysis. Fallback to standard clinical routing."
            }

        # 2. Construct the context-aware prompt
        prompt = f"""
        Patient Context: {json.dumps(patient_context)}

        Patient Message: "{user_message}"

        Analyze the patient's message. Determine if they are distressed, categorize the journey moment, and provide a safe, bounded response.
        """

        # 3. Call Gemini with strict JSON enforcement
        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=EmotionalIntervention,
                temperature=0.4  # Slightly higher temperature for natural empathy, but schema keeps it bounded
            )
        )

        return json.loads(response.text)



# 3. EXECUTION FOR TESTING

if __name__ == "__main__":
    # Ensure your API key is set in the environment or at the top of the file like we did in Module 2
    # os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY"

    gatekeeper = EmpathyGatekeeper()

    # Mock context from our FHIR vault
    mock_context = {
        "diagnosis": "Diminished Ovarian Reserve",
        "recent_event": "Received low AMH lab results"
    }

    # Test Scenario 1: High Anxiety regarding labs
    test_message = "I just saw my AMH results in the portal and I'm freaking out. It says 'low'. Does this mean I can never have a baby? I can't stop crying."

    print("Analyzing patient message...")
    result = gatekeeper.analyze_patient_message(test_message, mock_context)

    print("\n--- AI Concierge Emotional Intervention ---")
    print(json.dumps(result, indent=2))