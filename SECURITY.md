# Security notes

## Rotate these credentials NOW

The `.env` file in this working directory contained live credentials. Even
though `.env` is in `.gitignore`, treat all of these as compromised and rotate
them before continuing development:

1. **Slack incoming webhook** — `https://hooks.slack.com/services/T097ZH5JAGM/...`
   - Revoke at https://api.slack.com/apps → your app → Incoming Webhooks → delete the URL.
   - Generate a new webhook and put the new URL in your local `.env` only.

2. **OpenAI API key** — `sk-proj-bq0PCakeq766Pp...`
   - Revoke at https://platform.openai.com/api-keys.
   - Issue a new key, add to local `.env`. (Only `test.py` currently uses it.)

3. **Gemini API key** — `AIzaSyCXdOm57Pce_GEqgdtLLeE_mvxAFRzCkho`
   - Revoke at https://aistudio.google.com/app/apikey.
   - Issue a new key, add to local `.env`.

## After rotating

```bash
# Make sure .env never gets committed
git check-ignore -v .env   # should print: .gitignore:6:.env  .env

# If .env was ever committed historically, scrub it from the repo's git
# history. Coordinate with anyone else who has clones first.
pip install git-filter-repo
git filter-repo --path .env --invert-paths
git push --force origin main
```

## Going forward

- Use `.env.example` as the template for what variables are required.
- For deployed builds, prefer Streamlit secrets, GitHub Actions secrets, or a
  secrets manager (Doppler, AWS Secrets Manager, etc.) — never check secrets
  into the repo.
