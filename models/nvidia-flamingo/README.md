This folder creates a docker image that contains the NVIDIA Flamingo model.
It is meant to allow quick experimentation before official vLLM support is available fr Nvidia-Flamingo.
To build the image, run:
1. Copy the audio-flamingo-3 folder from (HuggingFace)[https://huggingface.co/spaces/nvidia/audio-flamingo-3/tree/main]  into this folder.
2. 
```bash
docker build -t nvidia-flamingo .
```
3. Run the image
