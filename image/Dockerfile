ARG VLLM_VERSION=v0.31.0
FROM vllm/vllm-openai:${VLLM_VERSION}

ENV VLLM_PORT=8000 \
    GPU_MEMORY_UTILIZATION=0.90

COPY model_store.py /opt/model_store.py
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS "http://localhost:${VLLM_PORT}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
