# Lambda container image for AWS Graviton (arm64).
FROM public.ecr.aws/lambda/python:3.12-arm64
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY inspectagent/ ${LAMBDA_TASK_ROOT}/inspectagent/
COPY eval/results/ ${LAMBDA_TASK_ROOT}/eval/results/
CMD ["inspectagent.app.handler"]
