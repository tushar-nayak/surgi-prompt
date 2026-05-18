#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${ROOT_DIR}/checkpoints"
mkdir -p "${CKPT_DIR}"

GDINO_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
SAM2_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

curl -L "${GDINO_URL}" -o "${CKPT_DIR}/groundingdino_swint_ogc.pth"
curl -L "${SAM2_URL}" -o "${CKPT_DIR}/sam2.1_hiera_large.pt"
