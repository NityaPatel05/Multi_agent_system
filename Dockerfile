# Main API service (architecture.md §7 deployment layer).
# Two stages: dependencies are built into a venv, then copied into a slim non-root runtime
# image (no compiler toolchain at runtime - every dependency ships wheels).
FROM python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f AS build

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.lock .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock


FROM python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f

RUN useradd --system --uid 10001 --create-home app
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY --chown=app:app research_langgraph/ ./research_langgraph/
COPY --chown=app:app eval/ ./eval/

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

USER app
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "research_langgraph.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
