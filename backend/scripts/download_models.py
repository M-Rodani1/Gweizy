#!/usr/bin/env python3
"""
Download ML models from GitHub Releases

This script downloads the spike detector models from GitHub Releases
during deployment. Used when Git LFS is not available (e.g., Render free tier).
"""

import os
import sys
import urllib.request
import hashlib

# GitHub release URL for model files
GITHUB_REPO = "M-Rodani1/Gweizy"
RELEASE_TAG = "models-v1.1"  # Updated to joblib format
BASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"

# Model files to download
# Note: v1.1 models are fallback models (~0.8MB each), trained on synthetic data
# They will be replaced with larger production models when trained on real data
MODELS = [
    {
        'name': 'spike_detector_1h.pkl',
        'url': f'{BASE_URL}/spike_detector_1h.pkl',
        'size_mb': 1  # Fallback model, ~0.8MB
    },
    {
        'name': 'spike_detector_4h.pkl',
        'url': f'{BASE_URL}/spike_detector_4h.pkl',
        'size_mb': 1  # Fallback model, ~0.8MB
    },
    {
        'name': 'spike_detector_24h.pkl',
        'url': f'{BASE_URL}/spike_detector_24h.pkl',
        'size_mb': 1  # Fallback model, ~0.8MB
    }
]


def download_file(url, destination, expected_size_mb=None):
    """Download a file with progress indication"""
    print(f"Downloading {os.path.basename(destination)}...")
    print(f"  URL: {url}")

    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = (downloaded / total_size) * 100 if total_size > 0 else 0
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')

        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print()  # New line after progress

        # Verify file size
        actual_size_mb = os.path.getsize(destination) / (1024 * 1024)
        print(f"  ✓ Downloaded: {actual_size_mb:.1f} MB")

        if expected_size_mb and abs(actual_size_mb - expected_size_mb) > 5:
            print(f"  ⚠️  Warning: Expected ~{expected_size_mb}MB, got {actual_size_mb:.1f}MB")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Main download function"""
    print("=" * 70)
    print("Downloading ML Models from GitHub Releases")
    print("=" * 70)

    # Determine model directories
    # Primary: /data/models (Railway persistent storage)
    # Fallback: backend/models/saved_models or models/saved_models
    model_dirs = []

    # Priority 1: Railway persistent storage
    if os.path.exists('/data'):
        persistent_dir = '/data/models'
        os.makedirs(persistent_dir, exist_ok=True)
        model_dirs.append(persistent_dir)
        print(f"Primary target: {persistent_dir} (persistent storage)")

    # Priority 2: Local development paths
    if os.path.exists('backend/models'):
        local_dir = 'backend/models/saved_models'
        os.makedirs(local_dir, exist_ok=True)
        model_dirs.append(local_dir)
    elif os.path.exists('models'):
        local_dir = 'models/saved_models'
        os.makedirs(local_dir, exist_ok=True)
        model_dirs.append(local_dir)

    if not model_dirs:
        print("Error: Cannot find models directory")
        sys.exit(1)

    print(f"Target directories: {model_dirs}")

    # Download each model to all target directories
    success_count = 0
    for model in MODELS:
        model_success = False

        for models_dir in model_dirs:
            destination = os.path.join(models_dir, model['name'])

            # Skip if already exists in this directory
            if os.path.exists(destination):
                size_mb = os.path.getsize(destination) / (1024 * 1024)
                print(f"✓ {model['name']} exists in {models_dir} ({size_mb:.1f} MB)")
                model_success = True
                continue

            print(f"\n{model['name']} -> {models_dir}:")
            if download_file(model['url'], destination, model['size_mb']):
                model_success = True

        if model_success:
            success_count += 1

    print("\n" + "=" * 70)
    print(f"Download Summary: {success_count}/{len(MODELS)} models ready")
    print("=" * 70)

    if success_count == len(MODELS):
        print("✓ All models downloaded successfully!")
        sys.exit(0)
    else:
        print("⚠️  Some models failed to download. Models may work with fallback predictions.")
        sys.exit(0)  # Don't fail build, just warn


if __name__ == '__main__':
    main()
