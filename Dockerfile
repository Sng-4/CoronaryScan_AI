# 1. Base Image: Python 3.11 (Required for modern libraries in your requirements.txt)
FROM python:3.11-slim

# 2. Optimization: Prevent Python from buffering logs (Crucial for Render logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. System Dependencies (Required for OpenCV/Pillow)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Security: Create a non-root user
RUN useradd -m -u 1000 appuser

# 5. Setup Workspace
WORKDIR /app

# 6. Install Dependencies (Fast Build)
# We copy requirements first to leverage caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Copy Application Code
COPY . .

# 8. Permissions: Ensure appuser owns the model file
RUN chown -R appuser:appuser /app

# 9. Switch to secure user
USER appuser

# 10. Expose Port (Documentation only)
EXPOSE 8000

# 11. Healthcheck (Satisfies Assignment "Extra Caveats")
# Uses the PORT environment variable to check the correct endpoint
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8000}/ || exit 1

# 12. Start Command
# CRITICAL FIX: Use shell format (no brackets) to allow variable expansion.
# This lets Render inject its dynamic PORT (usually 10000) or defaults to 8000.
CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}