#!/usr/bin/env python3
"""
Delete dataset versions from all locations.

Usage:
    python scripts/05_deployment/delete_dataset.py --version v3 --model-type tft
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment
project_root = Path(__file__).parent.parent.parent
env_file = project_root / '.env'
if env_file.exists():
    load_dotenv(env_file)

PROJECT_ID = os.getenv('GCP_PROJECT_ID')
REGION = os.getenv('GCP_REGION', 'us-central1')
GCS_BUCKET = f"{PROJECT_ID}-models" if PROJECT_ID else None


def delete_dataset(version: str, model_type: str, delete_local=True, delete_gcs=True, delete_managed=True):
    """Delete dataset from all locations."""
    import subprocess
    import shutil
    
    full_version = f"{model_type}/{version}"
    
    # 1. Delete local
    if delete_local:
        local_path = Path(f'data/datasets/{model_type}/{version}')
        if local_path.exists():
            shutil.rmtree(local_path)
        else:
            pass
    
    # 2. Delete GCS
    if delete_gcs:
        gcs_path = f"gs://{GCS_BUCKET}/datasets/{model_type}/{version}"
        cmd = f"gsutil -m rm -r {gcs_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        if result.returncode == 0:
            pass
        else:
            pass
    
    # 3. Delete Managed Dataset
    if delete_managed:
        try:
            from google.cloud import aiplatform
            aiplatform.init(project=PROJECT_ID, location=REGION)
            
            display_name = f"{model_type}-{version}-features-csv"
            datasets = aiplatform.TabularDataset.list(filter=f'display_name="{display_name}"')
            
            if datasets:
                datasets[0].delete()
            else:
        except Exception as e:
    
    # 4. Remove from registry
    registry_file = Path('datasets_registry.yaml')
    if registry_file.exists():
        with open(registry_file, 'r') as f:
            registry = yaml.safe_load(f) or {}
        
        if full_version in registry:
            del registry[full_version]
            with open(registry_file, 'w') as f:
                yaml.dump(registry, f, default_flow_style=False)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Delete dataset versions')
    parser.add_argument('--version', type=str, required=True, help='Version to delete (e.g., v3)')
    parser.add_argument('--model-type', type=str, default='tft', help='Model type')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')
    
    args = parser.parse_args()
    
    if not args.yes:
        response = input(f"\n⚠️  Delete {args.model_type}/{args.version}? [y/N]: ")
        if response.lower() != 'y':
            sys.exit(0)
    
    delete_dataset(args.version, args.model_type)
