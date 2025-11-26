# 1. Base Image
# We keep Python 3.11 because your dependencies (contourpy/click) require it.
FROM python:3.11-slim

# 2. Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. System Dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Create non-root user
RUN useradd -m -u 1000 appuser

# 5. Working Directory
WORKDIR /app

# 6. Install Dependencies using UV (Faster than Pip)
COPY requirements.txt .

# Step 6a: Install uv
RUN pip install uv

# Step 6b: Use uv to install requirements
# --system flag installs into the global python environment (standard for containers)
# --no-cache prevents storing huge cache files in the image layer
RUN uv pip install --system --no-cache -r requirements.txt

# 7. Copy Application Code
COPY . .

# 8. Permissions
RUN chown -R appuser:appuser /app && \
    mkdir -p /app/models && \
    chown -R appuser:appuser /app/models

# 9. Switch User
USER appuser

# 10. Expose Ports
EXPOSE 8000
EXPOSE 8501

# 11. Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

# 12. Start Command
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]