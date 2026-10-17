# One image for the api, the Granite worker and the simulator.
# Base: the L1CC Jupyter image (RHEL 10 UBI, Python 3.12 venv) which already has NNPA-enabled builds of
# PyTorch (torch_nnpa), XGBoost and llama.cpp (zDNN backend) for s390x.
ARG BASE=registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest
FROM ${BASE}

USER root
# `cryptography` publishes no s390x wheel, so build it here against the system OpenSSL 3.5
# (whose AES-GCM and SHA-256 run on CPACF). Rust is only needed for the build and is removed after.
RUN dnf -y install rust cargo openssl-devel \
 && /opt/venv/bin/pip install --no-cache-dir \
      "cryptography>=44,<51" "fastapi>=0.115,<1" "uvicorn>=0.30,<1" "PyJWT>=2.8,<3" \
 && dnf -y remove rust cargo openssl-devel \
 && dnf clean all && rm -rf /var/cache/dnf /root/.cargo

# writable locations for the non-root user (a fresh named volume inherits this ownership)
RUN mkdir -p /data /sim && chown 1000:1000 /data /sim

COPY --chown=1000:1000 src /app/src
COPY --chown=1000:1000 config /app/config

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1
WORKDIR /app
USER 1000
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "--factory", "scamshield.api.main:app_factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
