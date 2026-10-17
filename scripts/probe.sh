#!/usr/bin/env bash
# Environment probe. Runs ON the VM; prints evidence for docs/evidence/.
# Usage from the laptop:  scripts/remote.sh 'bash -s' < scripts/probe.sh > docs/evidence/probe-$(date +%Y%m%d-%H%M).txt
IMAGE="${IMAGE:-registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest}"
section() { printf '\n===== %s =====\n' "$1"; }

section "date / host";      date -u; hostname
section "architecture";     uname -a
section "os";               cat /etc/os-release
section "cpu";              nproc; lscpu | grep -E 'Architecture|Byte Order|Model name|Machine type|CPU\(s\)|Hypervisor'
section "machine type";     grep -E '^(Type|Model)' /proc/sysinfo
section "ai accelerator";   lscpu | grep -io 'nnpa' | head -1 || echo "nnpa NOT present"
section "cpu features";     grep -m1 '^features' /proc/cpuinfo
section "memory / disk";    free -h; df -h /
section "crypto (CPACF)";   openssl version
openssl speed -evp aes-256-gcm -seconds 2 2>/dev/null | tail -3
openssl speed -evp sha256 -seconds 2 2>/dev/null | tail -2
openssl list -signature-algorithms 2>/dev/null | grep -i -E 'ml-dsa|mldsa' || echo "no ML-DSA in this OpenSSL"
section "containers";       docker version --format '{{.Server.Version}}' 2>&1; docker compose version 2>&1; podman --version 2>&1

if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  section "jupyter image"
  docker image inspect "$IMAGE" --format 'id={{.Id}} size={{.Size}} created={{.Created}} arch={{.Architecture}}'
  docker run --rm --entrypoint bash "$IMAGE" -c '
    echo "--- python"; python -V; which python
    echo "--- key packages"; pip list 2>/dev/null | grep -iE "^(torch|torch.nnpa|xgboost|tensorflow |ibm-zdnn|llama|onnx |numpy|pandas|scikit-learn)"
    echo "--- gguf models"; ls -la /home/ibm-user/*.gguf 2>/dev/null || find / -name "*.gguf" -size +10M 2>/dev/null | head
    echo "--- torch_nnpa"; python - <<PY
import torch
print("torch", torch.__version__)
try:
    import torch_nnpa
    print("torch_nnpa", getattr(torch_nnpa, "__version__", "?"), "from", torch_nnpa.__file__)
    print("attrs", [a for a in dir(torch_nnpa) if not a.startswith("_")])
except Exception as e:
    print("torch_nnpa import failed:", repr(e))
PY
    echo "--- llama_cpp"; python -c "import llama_cpp, os; print(llama_cpp.__version__); print(os.listdir(os.path.join(os.path.dirname(llama_cpp.__file__), \"lib\")))"
  ' 2>&1
fi
