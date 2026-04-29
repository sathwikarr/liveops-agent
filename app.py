"""Deprecated — Streamlit UI was retired in Phase 15.

The website now runs on FastAPI:
    uvicorn web.server:app --reload    # → http://127.0.0.1:8000

This file is kept as a no-op stub so old `streamlit run app.py` invocations
print a clear message instead of erroring obscurely. Safe to delete.
"""
print("This project no longer uses Streamlit. Run the FastAPI website with:")
print("    uvicorn web.server:app --host 0.0.0.0 --port 8000")
print("or `docker run -p 8000:8000 liveops-agent`. See README.md.")
