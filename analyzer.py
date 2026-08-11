import requests
import ollama
import json
import re
import streamlit as st
from typing import List, Dict
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Mapping of extraction fields to their descriptions
FIELDS = {
    "data_collection": "Identify all explicitly mentioned types of personal data collected.",
    "data_usage": "Identify all explicitly stated purposes for using collected data.",
    "third_party_sharing": "State whether data is shared with third parties.",
    "user_rights": "List all explicitly stated user rights.",
    "security": "Extract all explicit statements about data security protections.",
    "key_risks": "Identify only explicit privacy risks implied by the policy."
}


def get_privacy_policy_text(url: str) -> str | None:
    """
    Fetches raw text from a privacy policy URL.
    - Downloads HTML
    - Parses it with BeautifulSoup
    - Returns clean text content
    """
    try:
        document = requests.get(url, timeout=10).text
        soup = BeautifulSoup(document, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    except Exception:
        return None


def split_text_into_chunks(text: str) -> List[str]:
    """
    Splits long privacy policy text into manageable chunks
    for LLM processing using LangChain's recursive splitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=100
    )
    return splitter.split_text(text)


def extract_json(raw: str):
    """
    Attempts to parse JSON from an LLM response.
    - First tries direct JSON parsing
    - If that fails, searches for a JSON-like block with regex
    """
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None


def analyze_section(text: str) -> Dict:
    """
    Sends a chunk of policy text to the LLM for structured extraction.
    Ensures:
    - Strict JSON output
    - No hallucinated or inferred information
    - Missing fields default to 'Not specified'
    """
    prompt = f"""
You are a strict privacy policy extraction engine.

Return ONLY valid JSON with:
- data_collection
- data_usage
- third_party_sharing
- user_rights
- security
- key_risks

Rules:
- Only explicit info
- No guessing
- If missing: "Not specified"

TEXT:
{text}
"""

    try:
        response = ollama.generate(
            model='phi4',
            prompt=prompt
        )

        parsed = extract_json(response.get("response", "")) or {}

        # Ensure all fields exist, defaulting to "Not specified"
        return {k: parsed.get(k, "Not specified") for k in FIELDS}

    except Exception:
        # If LLM fails, return error placeholders
        return {k: "Error" for k in FIELDS}


def merge_results(results: List[Dict]) -> Dict:
    """
    Merges extracted results from multiple text chunks.
    - Combines values for each field
    - Removes duplicates
    - Filters out errors and missing values
    """
    combined = {k: [] for k in FIELDS}

    for r in results:
        if not isinstance(r, dict):
            continue

        for k in FIELDS:
            val = r.get(k)

            # Skip missing or invalid values
            if not val or val in ["Not specified", "Error"]:
                continue
            if isinstance(val, dict):
                continue

            combined[k].append(val)

    final = {}

    for k, vals in combined.items():
        # Keep only string values
        safe_vals = [v for v in vals if isinstance(v, str)]
        # Remove duplicates while preserving order
        unique = list(dict.fromkeys(safe_vals))

        # Join into readable sentences
        final[k] = (
            " ".join(v.rstrip(".") + "." for v in unique)
            if unique else "Not specified"
        )

    return final


def to_bullets(text: str) -> str:
    """
    Converts a summary string into HTML bullet points.
    If no data exists, returns a simple 'Not specified' paragraph.
    """
    if not text or text == "Not specified":
        return "<p>Not specified</p>"

    # Split into sentences
    parts = [p.strip() for p in re.split(r"\.\s+|\.\n", text) if p.strip()]
    bullets = "".join(f"<li>{p}</li>" for p in parts)

    return f"<ul>{bullets}</ul>"


def main():
    """
    Streamlit UI:
    - Accepts a privacy policy URL
    - Fetches and processes the policy
    - Displays structured analysis in bullet format
    """
    st.title("Privacy Policy Analyzer")

    side = st.sidebar

    url = st.text_input("Enter URL of privacy policy:")

    if url:
        policy = get_privacy_policy_text(url)

        if not policy:
            st.error("Could not retrieve or parse the privacy policy.")
            return

        side.success("Policy Found")

        # Split into chunks for LLM processing
        chunks = split_text_into_chunks(policy)
        side.success("Policy Split")

        results = []

        # Analyze each chunk with spinner feedback
        with side.spinner("Analyzing chunks...", show_time=True):
            for i, chunk in enumerate(chunks, start=1):
                results.append(analyze_section(chunk))
                side.success(f"Analyzed chunk {i}/{len(chunks)}")

        # Merge chunk results
        final = merge_results(results)
        side.success("Results Merged")
        st.session_state['summary'] = final

        st.subheader("Analysis Results")

        summary = st.session_state['summary']

        # Prepare display categories
        answer = {
            'Category': [
                'Data Collection',
                'Data Usage',
                'Third Party Sharing',
                'User Rights',
                'Security',
                'Key Risks'
            ],
            'Summary': [
                to_bullets(summary.get("data_collection", "Not specified")),
                to_bullets(summary.get("data_usage", "Not specified")),
                to_bullets(summary.get("third_party_sharing", "Not specified")),
                to_bullets(summary.get("user_rights", "Not specified")),
                to_bullets(summary.get("security", "Not specified")),
                to_bullets(summary.get("key_risks", "Not specified"))
            ]
        }

        # Render results in Streamlit
        for category, summary_html in zip(answer['Category'], answer['Summary']):
            st.markdown(f"### {category}")
            st.markdown(summary_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()