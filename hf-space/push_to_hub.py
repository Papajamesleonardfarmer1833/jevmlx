"""
Create (or update) the Hugging Face Space for this demo and upload this folder.

Usage
-----

    # once, if you are not logged in yet:
    hf auth login

    # then:
    python push_to_hub.py

This creates the Space `<your-username>/parallel-constrained-decisions` if it does
not exist, uploads everything in this directory, and prints the Space URL.

Without an interactive login you can pass a write token through the environment:

    HF_TOKEN=hf_... python push_to_hub.py

The token is never stored by this script; it only reads the environment or the
token saved by `hf auth login`.

Requires: pip install huggingface_hub
"""

import argparse
import os
import sys
from pathlib import Path

SPACE_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_NAME = "parallel-constrained-decisions"

# Development-only artifacts that should not be uploaded.
IGNORE_PATTERNS = [
    "__pycache__/",
    "*.pyc",
    ".venv-dev/",
    ".git/",
    ".DS_Store",
]


def print_token_instructions():
    print(
        "\nNo usable Hugging Face token found.\n\n"
        "Option 1 (recommended): log in once with the Hugging Face CLI and re-run\n"
        "this script:\n\n"
        "    pip install -U huggingface_hub\n"
        "    hf auth login\n\n"
        "Option 2: create a write token at https://huggingface.co/settings/tokens\n"
        "and pass it through the environment:\n\n"
        "    HF_TOKEN=hf_... python push_to_hub.py\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create/update the demo Space and upload this folder."
    )
    parser.add_argument(
        "--repo-name",
        default=DEFAULT_REPO_NAME,
        help="Space repository name (default: %(default)s)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Space as private",
    )
    parser.add_argument(
        "--skip-hardware",
        action="store_true",
        help="Do not try to request ZeroGPU hardware automatically",
    )
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "huggingface_hub is not installed. Install it with:\n\n"
            "    pip install huggingface_hub\n"
        )
        return 1

    app_file = SPACE_DIR / "app.py"
    if not app_file.is_file():
        print("app.py was not found in {}.".format(SPACE_DIR))
        return 1

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or None
    )

    api = HfApi(token=token)
    try:
        username = api.whoami()["name"]
    except Exception as exc:
        print("Token check failed: {}".format(exc))
        print_token_instructions()
        return 1

    repo_id = "{}/{}".format(username, args.repo_name)
    print("Using account: {}".format(username))
    print("Space repository: {}".format(repo_id))

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=args.private,
            exist_ok=True,
        )
        print("Repository is ready (created now or already existed).")
    except Exception as exc:
        print("Could not create the Space: {}".format(exc))
        return 1

    print("Uploading {} ...".format(SPACE_DIR))
    try:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=str(SPACE_DIR),
            ignore_patterns=IGNORE_PATTERNS,
            commit_message="Upload parallel constrained decisions demo",
        )
    except Exception as exc:
        print("Upload failed: {}".format(exc))
        return 1
    print("Upload finished.")

    if not args.skip_hardware:
        try:
            api.request_space_hardware(repo_id=repo_id, hardware="zero-a10g")
            print("Requested ZeroGPU hardware (zero-a10g).")
        except Exception as exc:
            print("Could not request ZeroGPU hardware automatically: {}".format(exc))
            print(
                "Set it manually in the Space settings: Hardware -> ZeroGPU "
                "(zero-a10g)."
            )

    url = "https://huggingface.co/spaces/{}".format(repo_id)
    print("\nSpace URL: {}".format(url))
    print("Build logs: {}/logs".format(url))
    return 0


if __name__ == "__main__":
    sys.exit(main())
