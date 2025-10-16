FROM public.ecr.aws/lambda/python:3.11

WORKDIR /var/task

# copy requirements first for better layer cache
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# now copy the app code (context is already src/)
COPY . /var/task/

# Lambda handler entrypoint (FastAPI via Mangum)
CMD ["rag.api.app.handler"]