import streamlit as st
import json
import os
import pandas as pd
from orchestrator import PatientOrchestrator
from emotional_support import EmpathyGatekeeper
from omop_pipeline import OmopPipeline

# Ensure API Key is set for the session (Replace with your actual key if not in env)
if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = "PASTE_YOUR_ACTUAL_KEY_HERE"


# PAGE CONFIG & STATE

st.set_page_config(page_title="PtP Concierge", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant",
         "content": "Hi Jane. I'm your PtP Care Concierge. I see your AMH labs just came back. How are you feeling about everything today?"}
    ]


# SIDEBAR NAVIGATION

st.sidebar.title("Navigation")
view_mode = st.sidebar.radio("Select View:", ["Patient Portal (AI Concierge)", "Research Admin (OMOP ETL)"])


# VIEW 1: PATIENT PORTAL

if view_mode == "Patient Portal (AI Concierge)":
    st.title("Welcome to your Care Journey, Jane.")

    col1, col2 = st.columns([1, 1.5])

    # --- LEFT COLUMN: The Orchestrator (Journey Timeline) ---
    with col1:
        st.subheader("📋 Your Action Plan")
        st.info("The AI Orchestrator evaluates your FHIR records to generate this plan.")

        if st.button("Generate Care Plan"):
            with st.spinner("Analyzing clinical data..."):
                orchestrator = PatientOrchestrator()
                plan_json_str = orchestrator.evaluate_patient_state()
                plan_data = json.loads(plan_json_str)

                st.success(f"**Current Phase:** {plan_data['current_phase']}")

                if plan_data['missing_data']:
                    st.warning(f"**Missing Data Detected:** {', '.join(plan_data['missing_data'])}")

                for item in plan_data['action_items']:
                    with st.expander(f"{'✅' if item['status'] == 'completed' else '⏳'} {item['task']}"):
                        st.write(f"**Priority:** {item['priority'].title()}")
                        st.write(f"**Why this is needed:** {item['clinical_reason']}")

        # --- PROVIDER MATCHING UI ---
        st.divider()
        st.subheader("🩺 Provider Match")
        st.info("The AI Matching Agent ranks providers based on your diagnosis, location, and care stage.")

        if st.button("Find My Care Team"):
            with st.spinner("Scanning provider network..."):
                from provider_matching import ProviderMatchingAgent

                matcher = ProviderMatchingAgent()
                match_json_str = matcher.find_best_matches()
                match_data = json.loads(match_json_str)

                for provider in match_data['matches']:
                    st.metric(label=f"{provider['provider_name']} ({provider['specialty']})",
                              value=f"{provider['match_score']}% Match")
                    st.write(f"**Why:** {provider['reason']}")
                    st.caption(f"**Next Step:** {provider['next_step']}")
                    st.write("---")

    # --- RIGHT COLUMN: The Empathy Interceptor (Chat) ---
    with col2:
        st.subheader("💬 AI Care Concierge")

        # Display Chat History
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Chat Input
        if user_input := st.chat_input("Type your message here..."):
            # 1. Append user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            # 2. Process with Empathy Gatekeeper
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    gatekeeper = EmpathyGatekeeper()
                    # Mock context based on our FHIR data
                    context = {"diagnosis": "Diminished Ovarian Reserve", "recent_event": "Low AMH Lab"}

                    response_data = gatekeeper.analyze_patient_message(user_input, context)

                    if response_data.get("requires_escalation"):
                        st.error(
                            "🚨 **CLINICAL ESCALATION TRIGGERED** 🚨\n\nI am pinging the on-call nurse immediately. Please call 911 if this is a medical emergency.")
                    elif response_data.get("is_distressed"):
                        st.write(response_data["empathetic_response"])
                        st.info(f"**Suggested Resource:** {response_data['recommended_resource']}")
                    else:
                        st.write(response_data["empathetic_response"])

            # 3. Append assistant message
            st.session_state.chat_history.append({"role": "assistant",
                                                  "content": response_data.get("empathetic_response",
                                                                               "I am escalating this to a human care coordinator.")})


# VIEW 2: RESEARCH ADMIN (OMOP ETL)

elif view_mode == "Research Admin (OMOP ETL)":
    st.title(" Data Engineering: OMOP ETL Pipeline")
    st.write(
        "This dashboard simulates the backend transformation of operational FHIR data into IRB-compliant OMOP tables for clinical research.")

    if st.button("Run ETL Pipeline & Apply Safe Harbor Rules"):
        with st.spinner("Processing files..."):
            pipeline = OmopPipeline()
            person_df = pipeline.process_person_table()
            measurement_df = pipeline.process_measurement_table()

            st.subheader("De-Identified `PERSON` Table")
            st.write("Notice that names and exact dates have been stripped. Gender is mapped to concept `8532`.")
            st.dataframe(person_df)

            st.subheader("`MEASUREMENT` Table")
            st.write(
                "Notice that AMH was pulled, but FSH was skipped because its FHIR status was pending ('registered').")
            st.dataframe(measurement_df)