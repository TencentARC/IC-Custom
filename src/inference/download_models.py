import os
import argparse
from huggingface_hub import snapshot_download, hf_hub_download


args = argparse.ArgumentParser()
args.add_argument("--hf_token", type=str, default=None)
args.add_argument("--local_dir", type=str, default="models")
args = args.parse_args()


if __name__ == "__main__":
    if args.hf_token is not None:
        os.environ["HF_TOKEN"] = args.hf_token

    hf_hub_download(
        repo_id="black-forest-labs/FLUX.1-Fill-dev",
        filename="flux1-fill-dev.safetensors",
        local_dir=args.local_dir,
        resume_download=True,
    )
    hf_hub_download(
        repo_id="black-forest-labs/FLUX.1-Fill-dev",
        filename="ae.safetensors",
        local_dir=args.local_dir,
        resume_download=True,
    )
    hf_hub_download(
        repo_id="black-forest-labs/FLUX.1-Redux-dev",
        filename="flux1-redux-dev.safetensors",
        local_dir=args.local_dir,
        resume_download=True,
    )

    snapshot_download(
        repo_id="TencentARC/IC-Custom",
        local_dir=os.path.join(args.local_dir, "ic-custom"),
        resume_download=True,
    )

    snapshot_download(
        repo_id="openai/clip-vit-large-patch14",
        local_dir=os.path.join(args.local_dir, "clip-vit-large-patch14"),
        resume_download=True,
        ignore_patterns=["*.h5", "tf_model.h5"], # Ignore TensorFlow model files
    )
    snapshot_download(
        repo_id="google/t5-v1_1-xxl",
        local_dir=os.path.join(args.local_dir, "t5-v1_1-xxl"),
        resume_download=True,
        ignore_patterns=["*.h5", "tf_model.h5"], # Ignore TensorFlow model files
    )
    snapshot_download(
        repo_id="google/siglip-so400m-patch14-384",
        local_dir=os.path.join(args.local_dir, "siglip-so400m-patch14-384"),
        resume_download=True,
        ignore_patterns=["*.h5", "tf_model.h5"], # Ignore TensorFlow model files
    )

    print("Done, The models are saved in ", args.local_dir)