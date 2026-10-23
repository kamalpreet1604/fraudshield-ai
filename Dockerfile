# Recipe for the FraudShield API box (image).
# Build:  docker build -t fraudshield-api .
# Run:    docker run --rm -p 8080:8080 fraudshield-api
# Then open http://localhost:8080/docs

# 1. start from a small official box that already has Python
#    (same Python version the model was trained with)
FROM python:3.14-slim

# 2. bring in uv, the package installer, from its official box
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /bin/uv

# 3. the folder inside the box where everything lives
WORKDIR /app

# 4. shopping list first, then install, so Docker can reuse this slow step
#    when only the code changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project

# 5. our code and the trained model
COPY src ./src
COPY models ./models

# 6. the door the API listens on (Cloud Run expects 8080)
EXPOSE 8080

# 7. what happens every time the box starts
CMD ["uv", "run", "--no-sync", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
