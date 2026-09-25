FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-deps .

COPY app/ app/
COPY models/ models/
COPY .streamlit/ .streamlit/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen(f'http://localhost:{__import__(\"os\").environ[\"PORT\"]}/_stcore/health')"

# Shell form so $PORT expands (Hugging Face Spaces uses 7860, Render/Railway set PORT).
CMD streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
