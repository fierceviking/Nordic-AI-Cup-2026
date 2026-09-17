FROM python:3.11-slim

# opencv needs libGL and libglib even in headless use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Your model and the server. The sample data under src/ is not copied: it is
# 400 MB and only local_evaluator.py needs it. model/ holds the trained
# weights and is required at import time by solution.py.
COPY dtos.py utils.py example.py solution.py api.py ./
COPY model/ ./model/

EXPOSE 9053

CMD ["python", "api.py"]
