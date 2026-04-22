# Launching vLLM Endpoints

This folder contains a Kubernetes deployment example ([k8.yaml](k8.yaml)) and guidance for launching vLLM endpoints that can serve Large Audio-Language Models (LALMs).

## Deployment Approaches

You can use either of the following approaches:

- **Local**: Run a vLLM server on your workstation or VM (ideal for development)
- **Kubernetes**: Deploy the provided [k8.yaml](k8.yaml) to a GPU-capable cluster

## Important Notes

- **Security**: Do NOT commit real secrets (Hugging Face tokens) into source control. Use Kubernetes Secrets or environment variables stored securely.
- **Image Configuration**: The [k8.yaml](k8.yaml) file uses a placeholder image (`<YOUR_VLLM_IMAGE_WITH_AUDIO_DEPENDANCIES_INSTALLED>`). Replace this with an image that has required audio dependencies (ffmpeg, soundfile, librosa, torchaudio, and any model-specific libraries) before applying.
- **Port Configuration**: The example exposes ports 8000-8007. If you only need a single instance, reduce the number of containers/ports in the Pod accordingly.



## Useful Resources

### vLLM Documentation
- [Overview & Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/)
- [CLI `serve` Documentation](https://docs.vllm.ai/en/latest/cli/serve/)
- [Kubernetes Deployment Guide](https://docs.vllm.ai/en/latest/deployment/k8s/)

### Audio & Multimodal Support
- [Audio Assets API](https://docs.vllm.ai/en/latest/api/vllm/assets/audio/)
- [Audio-Language Offline Inference Example](https://docs.vllm.ai/en/latest/examples/offline_inference/audio_language/)

These audio-specific resources describe how vLLM handles audio assets, required dependencies, and example code for audio-language workflows.


---

## A. Local Development Setup

### 1. Prerequisites

- GPU node or a machine with a compatible PyTorch/CUDA setup (or CPU-only for small models)
- Python 3.10+ (virtual environment recommended)
- Hugging Face token with model access, set in `HUGGING_FACE_HUB_TOKEN`

### 2. Installation

Install vLLM and required audio dependencies:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install vLLM
pip install vllm --upgrade
```

**macOS (Homebrew):**
```bash
brew install ffmpeg libsndfile
pip install soundfile librosa torchaudio
```

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1
pip install soundfile librosa torchaudio
```

### 3. Start the Server

The vLLM CLI provides a `serve` entrypoint that starts an OpenAI-compatible HTTP server:

```bash
# Export your Hugging Face token
export HUGGING_FACE_HUB_TOKEN="<YOUR_HF_TOKEN>"

# Start the vLLM server
vllm serve Qwen/Qwen3-Omni-30B-A3B-Thinking --port 8000 --host 0.0.0.0
```

**Additional Notes:**
- Use `--api-key` or set `VLLM_API_KEY` if you want the server to require an API key
- Many LALMs need additional Python packages or system libraries. Commonly required: `soundfile`, `librosa`, `torchaudio`, and system `ffmpeg`/`libsndfile`. Check the model's Hugging Face page and vLLM audio documentation for specific requirements
- For GPU acceleration, ensure a compatible PyTorch/CUDA combination is installed (or use vLLM Docker images with prebuilt CUDA support)

### 4. Configure Your Run Config

Update your run config to point to the local server:

```yaml
# Example run_configs entry
# For OpenAI-compatible API calls use endpoints like /v1/completions or /v1/chat/completions
url: "http://localhost:8000/v1/completions"
```





---

## B. Kubernetes Deployment

### Overview

The provided [k8.yaml](k8.yaml) configuration:

- Launches a single Pod template containing multiple vLLM containers (ports 8000-8007)
- Each container runs the same model and listens on a distinct port
- Exposes the Pod ports on cluster nodes via a `NodePort` Service

### Pre-Deployment Checklist

#### 1. Update Container Image

Replace the placeholder image `<YOUR_VLLM_IMAGE_WITH_AUDIO_DEPENDANCIES_INSTALLED>` in [k8.yaml](k8.yaml) with an image that includes:
- vLLM installed
- Python audio libraries: `soundfile`, `librosa`, `torchaudio`, etc.
- System binaries: `ffmpeg` and `libsndfile`

#### 2. Configure Secrets

Create a Kubernetes Secret for your Hugging Face token:

```bash
kubectl -n <namespace> create secret generic hf-token \
  --from-literal=HUGGING_FACE_HUB_TOKEN='<YOUR_HF_TOKEN>'
```

Then update the container environment in [k8.yaml](k8.yaml) to use `valueFrom.secretKeyRef` instead of a plain `value`.

#### 3. Verify Cluster Requirements

- **GPU Support**: Ensure GPU-enabled nodes and drivers are available (matching the image/CUDA version)
- **Scheduler**: If using Run:AI or a custom scheduler, verify `schedulerName` matches your cluster configuration. Remove or edit if not applicable.

### Deployment Steps

Apply the configuration:

```bash
# Apply the Kubernetes manifest
kubectl apply -f vllm_configs/k8.yaml

# Monitor the rollout
kubectl -n <namespace> rollout status deployment/infer-qwen3-omni
kubectl -n <namespace> get pods -l app=infer-qwen3-omni
```

### Accessing the Service

The Service uses `NodePort` type. To find assigned node ports:

```bash
kubectl -n <namespace> get svc infer-qwen3-omni-service -o wide
```

Access the service using `http://<node-ip>:<nodePort>` for ports 8000-8007.

For production environments, consider exposing via `LoadBalancer` or an Ingress controller.

### Troubleshooting

- **Check Logs**: `kubectl -n <namespace> logs <pod> -c deployment0` (replace with actual container name)
- **Model Loading Issues**: Verify `HUGGING_FACE_HUB_TOKEN`, check CUDA/PyTorch compatibility, and ensure `--trust_remote_code` is only set for trusted model repositories

> [!NOTE]
> For models requiring LoRA adapters (e.g., Phi-4 Multimodal), ensure LoRA-related flags and paths are correctly configured in the command arguments. See [phi4_k8.yaml](phi4_k8.yaml) for an example.