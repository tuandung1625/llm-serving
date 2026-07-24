FROM vllm/vllm-openai:v0.22.1

ENV PYTHONUNBUFFERED=1 \
    VLLM_USAGE_STATS=0 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /workspace
COPY scripts/healthcheck.py /opt/llm-serving-baseline/healthcheck.py

EXPOSE 8000

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model=/model", "--served-model-name=LFM2.5-1.2B-Instruct", "--host=0.0.0.0", "--port=8000", "--max-model-len=32768", "--gpu-memory-utilization=0.95", "--tensor-parallel-size=1", "--enable-prefix-caching"]

