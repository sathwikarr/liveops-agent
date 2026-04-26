# Deployment

Three options, pick one.

## 1. Local — macOS launchd (background loop only)

Runs the autonomous detection loop as a user-level macOS service. The Streamlit
UI is started separately when you want to look at the dashboard.

    # 1. Edit the three REPLACE_ME paths in com.liveops.agent.plist to your repo root + venv python
    # 2. Drop it into LaunchAgents and load it:
    cp deploy/com.liveops.agent.plist ~/Library/LaunchAgents/
    launchctl load -w ~/Library/LaunchAgents/com.liveops.agent.plist

    # Tail logs
    tail -f /tmp/liveops-agent.out /tmp/liveops-agent.err

    # Stop
    launchctl unload -w ~/Library/LaunchAgents/com.liveops.agent.plist

The plist sets `KeepAlive=true` and `ThrottleInterval=10`, so launchd restarts
the loop ten seconds after any crash.

## 2. Docker — single container (UI only or loop only)

    # Build once
    docker build -t liveops-agent .

    # UI on http://localhost:8501
    docker run --rm -p 8501:8501 \
      --env-file .env \
      -v "$PWD/data:/app/data" \
      -v "$PWD/user_data:/app/user_data" \
      liveops-agent

    # Or the headless loop
    docker run --rm \
      --env-file .env \
      -v "$PWD/data:/app/data" \
      -v "$PWD/user_data:/app/user_data" \
      liveops-agent python auto_agent.py

## 3. Docker Compose — UI + loop together (recommended)

    docker compose up --build       # builds, runs both services
    docker compose logs -f auto     # tail the loop
    docker compose down

The two services share `./data` (SQLite WAL handles concurrent reads from the
dashboard while the loop writes), so signups in the UI immediately work for the
auto-agent and vice versa.

## Secrets

Never bake `.env` or `streamlit/secrets.toml` into an image — both are listed in
`.dockerignore`. Pass them at runtime via `--env-file .env` or compose's
`env_file:`. See `SECURITY.md` for key rotation steps.
