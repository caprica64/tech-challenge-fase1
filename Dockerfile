# AWS Lambda base image — includes the Lambda Runtime Interface Client
FROM public.ecr.aws/lambda/python:3.12

# Install dependencies into the Lambda task root (layer caching: copy first, then install)
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy application source and saved models
COPY src/    ${LAMBDA_TASK_ROOT}/src/
COPY models/ ${LAMBDA_TASK_ROOT}/models/
COPY main.py ${LAMBDA_TASK_ROOT}/

# Handler format: <module>.<function>
# Lambda invokes handler(event, context) via Mangum
CMD ["main.handler"]
