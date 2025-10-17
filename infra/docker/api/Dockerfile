FROM public.ecr.aws/lambda/python:3.11

WORKDIR ${LAMBDA_TASK_ROOT}

# copy requirements first for better layer cache
COPY src/requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -m pip install pydantic-core --platform manylinux2014_aarch64 -t . --only-binary=:all:

# now copy the app code (context is already src/)
COPY . .

# Lambda handler entrypoint (FastAPI via Mangum)
CMD ["rag.api.app.handler"]