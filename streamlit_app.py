"""Streamlit uploader for Maritime AI FastAPI."""

import os
import json

import httpx
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(
    page_title="Maritime AI",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Maritime AI — Certificate Auto-Fill")

st.caption(
    "Upload a maritime certificate PDF, classify it, "
    "and extract certificate information using "
    "PaddleOCR and optional Qwen2.5-VL."
)

# ---------------------------------------
# Sidebar settings
# ---------------------------------------

url = st.sidebar.text_input(
    "FastAPI endpoint",
    os.getenv(
        "FASTAPI_URL",
        "http://127.0.0.1:8000"
    )
)

use_qwen = st.sidebar.checkbox(
    "Enable Qwen2.5-VL (Ollama)",
    value=True,
    help="Requires Ollama and Qwen2.5-VL installed"
)

st.sidebar.info(
    "Uploaded PDFs are processed through your "
    "configured FastAPI server."
)

# ---------------------------------------
# Check FastAPI connection
# ---------------------------------------

try:
    response = httpx.get(
        url.rstrip("/") + "/health",
        timeout=3,
        trust_env=False
    )

    response.raise_for_status()
    health = response.json()

    st.sidebar.success(
        "API connected: " +
        health.get("status", "unknown")
    )

except Exception:
    st.sidebar.warning(
        "Start FastAPI on port 8000 first."
    )

# ---------------------------------------
# Upload certificate PDF
# ---------------------------------------

upload = st.file_uploader(
    "Choose a maritime certificate PDF",
    type=["pdf"]
)

if upload and st.button(
    "Extract certificate",
    type="primary"
):

    with st.spinner("Analyzing PDF..."):

        try:
            response = httpx.post(
                url.rstrip("/") +
                "/api/v1/certificates/extract",

                files={
                    "file": (
                        upload.name,
                        upload.getvalue(),
                        "application/pdf"
                    )
                },

                data={
                    "use_ollama": str(use_qwen).lower()
                },

                timeout=600,
                trust_env=False
            )

            response.raise_for_status()

            result = response.json()

            st.session_state["result"] = result

        except Exception as exc:
            st.error(
                "Request failed: " + str(exc)
            )

# ---------------------------------------
# Display extracted certificate data
# ---------------------------------------

if "result" in st.session_state:

    result = st.session_state["result"]

    data = result.get("data") or {}

    if result.get("success"):

        st.success(
            "Certificate processing completed. "
            "Please verify the extracted values."
        )

    else:
        st.error(
            "Certificate processing was unsuccessful."
        )

    # Certificate overview
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Certificate",
        data.get("certificate_name") or "Unknown"
    )

    c2.metric(
        "Start Date",
        data.get("start_date") or "Not found"
    )

    c3.metric(
        "End Date",
        data.get("end_date") or "Not found"
    )

    # ---------------------------------------
    # Extracted fields
    # ---------------------------------------

    st.subheader("Extracted Information")

    fields = {
        "Certificate Type": data.get(
            "certificate_type"
        ),
        "Certificate Name": data.get(
            "certificate_name"
        ),
        "Expiration": data.get(
            "expiration"
        ),
        "Category": data.get(
            "category"
        ),
        "Start Date": data.get(
            "start_date"
        ),
        "End Date": data.get(
            "end_date"
        ),
        "License Number": data.get(
            "license_number"
        )
    }

    st.dataframe(
        [
            {
                "Field": key,
                "Value": (
                    str(value)
                    if value is not None
                    else "null"
                )
            }
            for key, value in fields.items()
        ],
        hide_index=True,
        use_container_width=True
    )

    # ---------------------------------------
    # Complete JSON response
    # ---------------------------------------

    st.subheader("JSON Response")

    with st.expander(
        "Complete JSON response",
        expanded=True
    ):
        st.json(result)

    # ---------------------------------------
    # Download JSON
    # ---------------------------------------

    st.download_button(
        label="Download JSON",
        data=json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        ),
        file_name="maritime_certificate.json",
        mime="application/json"
    )

    st.warning(
        "AI/OCR extraction can contain errors. "
        "Verify certificate name, dates, "
        "and license number against the original PDF."
    )