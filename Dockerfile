FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md requirements-cpu.lock ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements-cpu.lock && pip install --no-deps .
COPY configs ./configs
COPY tests ./tests
CMD ["python", "-m", "mjsrl", "smoke", "--steps", "10"]
