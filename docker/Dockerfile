# ── Base image ──
# We use Python 3.11 slim — smaller image size, faster to build
FROM python:3.11-slim

# ── Set working directory ──
# All commands will run from /app inside the container
WORKDIR /app

# ── Copy requirements first ──
# Docker builds in layers — copying requirements before code means
# pip install is cached and won't re-run unless requirements.txt changes
COPY requirements.txt .

# ── Install dependencies ──
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy the rest of the project ──
COPY . .

# ── Expose ports ──
# 8000 = FastAPI, 8501 = Streamlit
EXPOSE 8000
EXPOSE 8501

# ── Default command ──
# Runs FastAPI by default
# To run Streamlit instead, override this when running the container
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
