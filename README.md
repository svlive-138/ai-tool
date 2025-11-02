# AI Grammar Correction Tool

This is a Streamlit app that uses a T5-based grammar correction model to correct text and provide an audio playback of corrected text.

Key points
- The app uses browser Web Speech API for TTS (no server-side audio) to keep deployment simple and responsive.
- Large model weights are not included in the repo. Use Hugging Face-hosted models or configure your own model.

Quick start (local)
1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
python -m nltk.downloader punkt
```

3. Run the app:

```powershell
streamlit run spelling_correction_app.py
```

Deploying to Streamlit Cloud
- Push this repo to GitHub (do not include large model files or secrets).
- On https://share.streamlit.io create a new app and point it to this repository and branch.
- Streamlit Cloud will install packages from `requirements.txt`.

Notes and troubleshooting
- Browser TTS uses the Web Speech API. For immediate playback, the browser may require a user interaction. If the speech doesn't start automatically, interact with the page (click or focus) and try again.
- If you want server-side audio (pyttsx3), add it back and pin the package in `requirements.txt`, but note server-side audio will play on the host, not in the browser.
- If the model fails to load due to transformer/torch versions, install compatible wheels (see `transformers` and `torch` documentation). Streamlit Cloud provides CPU-only runtimes; consider smaller/faster models for deployment.

Security
- Do not commit API keys, tokens, or credentials. Use Streamlit Secrets to store private values when deploying.
