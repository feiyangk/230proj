#!/usr/bin/env python3
"""
Submit single training job to Vertex AI

Usage:
  # Use named job from vertex.yaml
  python submit_job.py --job fincast-gpu
  
  # Override specific parameters
  python submit_job.py --job fincast-gpu --dataset-version v4
  
  # Specify all parameters manually (legacy mode)
  python submit_job.py --dataset-version v3 --model-type decoder_transformer --profile gpu-t4
"""

import os
import yaml
from pathlib import Path
from google.cloud import aiplatform
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
project_root = Path(__file__).parent.parent.parent
env_file = project_root / '.env'

if not env_file.exists():
    raise FileNotFoundError(f".env file not found at {env_file}")

load_dotenv(env_file)

# Load vertex.yaml config
vertex_config_path = project_root / 'configs' / 'vertex.yaml'
with open(vertex_config_path, 'r') as f:
    VERTEX_CONFIG = yaml.safe_load(f)

# Configuration from environment (with fallbacks to vertex.yaml)
PROJECT_ID = os.getenv('GCP_PROJECT_ID')
REGION = os.getenv('GCP_REGION', 'us-central1')

if not PROJECT_ID:
    raise ValueError(
        f"GCP_PROJECT_ID not set in .env file ({env_file}). "
        "Please set it in .env or use: export GCP_PROJECT_ID=your-project-id"
    )

GCS_BUCKET = VERTEX_CONFIG['project'].get('gcs_bucket', f"{PROJECT_ID}-models")
IMAGE_URI = VERTEX_CONFIG['project'].get('image_uri', f"gcr.io/{PROJECT_ID}/inflation-predictor:latest")


def submit_training_job(
    job_name=None,
    machine_type='n1-standard-4',  # N1 supports GPUs
    accelerator_type='NVIDIA_TESLA_T4',  # T4 GPU enabled by default
    accelerator_count=1,  # 1 GPU
    use_spot=True,  # Use spot instances for faster provisioning & lower cost
    dataset_version=None,  # Dataset version (e.g., 'v1', 'v2')
    model_type='tft',  # Model type (e.g., 'tft', 'lstm', 'transformer')
    **hyperparameters
):
    """
    Submit a custom training job to Vertex AI.
    
    Args:
        job_name: Name for the training job
        machine_type: GCE machine type
        accelerator_type: GPU type (optional)
        accelerator_count: Number of GPUs
        use_spot: Use spot (preemptible) instances
        dataset_version: Dataset version (e.g., 'v1', 'v2'). If not provided, generates from BigQuery.
        model_type: Model type (e.g., 'tft', 'lstm', 'transformer')
        **hyperparameters: Model hyperparameters to pass
    """
    
    if job_name is None:
        job_name = f"model-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Initialize Vertex AI with staging bucket
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f'gs://{GCS_BUCKET}'
    )
    
    # Build args list
    args = [
        f'--gcs_bucket={GCS_BUCKET}',
        f'--job_name={job_name}',
        f'--model_type={model_type}',  # Pass model type to training wrapper
    ]
    
    # Add dataset version if provided
    # Construct full dataset path: model_type/version (e.g., 'tft/v1')
    if dataset_version:
        full_dataset_version = f"{model_type}/{dataset_version}"
        args.append(f'--dataset_version={full_dataset_version}')
    
    for key, value in hyperparameters.items():
        args.append(f'--{key}={value}')
    
    if dataset_version:
        full_dataset_version = f"{model_type}/{dataset_version}"
    else:
        full_dataset_version = None
    # Create custom job
    machine_spec = {
        'machine_type': machine_type,
    }
    
    if accelerator_type and accelerator_count > 0:
        machine_spec['accelerator_type'] = accelerator_type
        machine_spec['accelerator_count'] = accelerator_count
    
    # Build worker pool spec
    worker_pool_spec = {
        'machine_spec': machine_spec,
        'replica_count': 1,
        'container_spec': {
            'image_uri': IMAGE_URI,
            'command': ['python', 'scripts/05_deployment/train_vertex.py'],
            'args': args,
        },
    }
    
    # Create labels for easy identification in Experiments UI
    # Labels must be lowercase, alphanumeric, hyphens, underscores
    labels = {
        'model_type': model_type.lower().replace('_', '-'),
        'job_name': job_name.lower().replace('_', '-'),
    }
    if dataset_version:
        labels['dataset_version'] = dataset_version.lower().replace('/', '-').replace('_', '-')
    
    # Try to extract date range from model config
    try:
        # Map model_type to config file
        config_map = {
            'tft': 'configs/model_tft_config.yaml',
            'lstm': 'configs/model_lstm_config.yaml',
            'decoder_transformer': 'configs/model_decoder_config.yaml',
        }
        config_path = config_map.get(model_type)
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                model_config = yaml.safe_load(f)
            
            start_date = model_config.get('data', {}).get('start_date', '')
            end_date = model_config.get('data', {}).get('end_date', '')
            
            if start_date:
                # GCP labels: lowercase, alphanumeric, hyphens, underscores only
                labels['start_date'] = start_date.replace('/', '-')
            if end_date:
                labels['end_date'] = end_date.replace('/', '-')
    except Exception as e:
        # Non-critical, continue without date labels
    
    for key, value in labels.items():
    
    job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=[worker_pool_spec],
        labels=labels,
    )
    
    # Get or create TensorBoard instance
    tensorboard_resource_name = None
    try:
        tensorboards = aiplatform.Tensorboard.list(filter=f'display_name="tensorboard-{PROJECT_ID}"')
        
        if tensorboards:
            tensorboard = tensorboards[0]
            tensorboard_resource_name = tensorboard.resource_name
        else:
            tensorboard = aiplatform.Tensorboard.create(
                display_name=f"tensorboard-{PROJECT_ID}",
                project=PROJECT_ID,
                location=REGION,
            )
            tensorboard_resource_name = tensorboard.resource_name
    except Exception as e:
        tensorboard_resource_name = None
    
    # Submit job
    if use_spot:
    
    # Prepare job run parameters
    run_params = {'sync': False}
    
    # Add TensorBoard and service account if TensorBoard is configured
    if tensorboard_resource_name:
        # Get project number for default compute service account
        # This SA already has necessary GCS and Vertex AI permissions
        import subprocess
        result = subprocess.run(
            ['gcloud', 'projects', 'describe', PROJECT_ID, '--format=value(projectNumber)'],
            capture_output=True, text=True, check=True
        )
        project_number = result.stdout.strip()
        service_account = f"{project_number}-compute@developer.gserviceaccount.com"
        
        run_params['tensorboard'] = tensorboard_resource_name
        run_params['service_account'] = service_account
    else:
    
    job.run(**run_params)
    
    # Wait a moment for job to be created
    import time
    time.sleep(2)
    
    
    # Job properties may not be available immediately after async submission
    try:
    except (RuntimeError, AttributeError):
    
    try:
    except (RuntimeError, AttributeError):
    
    
    return job


if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Submit Vertex AI training job',
        epilog='Examples:\n'
               '  # Use named job from vertex.yaml\n'
               '  python submit_job.py --job fincast-gpu\n'
               '\n'
               '  # Override parameters\n'
               '  python submit_job.py --job fincast-gpu --dataset-version v4\n'
               '\n'
               '  # Manual mode\n'
               '  python submit_job.py --profile gpu-t4 --model-type decoder_transformer --dataset-version v3\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Named job (easiest way)
    parser.add_argument('--job', type=str, default=None,
                       help='Named job from vertex.yaml (e.g., fincast-gpu, fincast-cpu, test-cpu)')
    
    # Individual overrides
    parser.add_argument('--profile', type=str, default=None,
                       help='Machine profile (cpu, cpu-large, gpu-t4, gpu-v100, gpu-a100)')
    parser.add_argument('--dataset-version', type=str, default=None,
                       help='Dataset version to use (e.g., v3, v4)')
    parser.add_argument('--model-type', type=str, default=None,
                       help='Model type (e.g., decoder_transformer, tft, lstm)')
    parser.add_argument('--job-name', type=str, default=None,
                       help='Custom job name (auto-generated if not provided)')
    parser.add_argument('--machine-type', type=str, default=None,
                       help='GCE machine type (overrides profile)')
    parser.add_argument('--accelerator', type=str, default=None,
                       help='GPU type (overrides profile)')
    parser.add_argument('--accelerator-count', type=int, default=None,
                       help='Number of GPUs (overrides profile)')
    parser.add_argument('--gcs-bucket', type=str, default=None,
                       help='GCS bucket name (overrides vertex.yaml)')
    parser.add_argument('--wait', action='store_true',
                       help='Wait and monitor job status (default: exit immediately)')
    args = parser.parse_args()
    
    # Build job configuration from vertex.yaml + CLI args
    job_config = {}
    
    # Start with defaults from vertex.yaml
    if args.job:
        # Named job from vertex.yaml
        if args.job not in VERTEX_CONFIG['jobs']:
            for job_name, job_def in VERTEX_CONFIG['jobs'].items():
                profile = VERTEX_CONFIG['profiles'][job_def['profile']]
            exit(1)
        
        # Load named job config
        named_job = VERTEX_CONFIG['jobs'][args.job]
        job_config.update(named_job)
        
        # Load profile settings
        profile_name = named_job['profile']
        profile = VERTEX_CONFIG['profiles'][profile_name]
        job_config.update({
            'machine_type': profile['machine_type'],
            'accelerator_type': profile['accelerator_type'],
            'accelerator_count': profile['accelerator_count'],
        })
        
    else:
        # Manual configuration with defaults from vertex.yaml
        job_config = {
            'model_type': args.model_type or VERTEX_CONFIG['defaults']['model_type'],
            'dataset_version': args.dataset_version or VERTEX_CONFIG['defaults']['dataset_version'],
            'job_name': args.job_name,
        }
        
        # Use profile if specified
        if args.profile:
            if args.profile not in VERTEX_CONFIG['profiles']:
                for prof_name, prof_def in VERTEX_CONFIG['profiles'].items():
                exit(1)
            
            profile = VERTEX_CONFIG['profiles'][args.profile]
            job_config.update({
                'machine_type': profile['machine_type'],
                'accelerator_type': profile['accelerator_type'],
                'accelerator_count': profile['accelerator_count'],
            })
        else:
            # Default to CPU profile
            profile = VERTEX_CONFIG['profiles']['cpu']
            job_config.update({
                'machine_type': profile['machine_type'],
                'accelerator_type': profile['accelerator_type'],
                'accelerator_count': profile['accelerator_count'],
            })
    
    # Override with CLI arguments (highest priority)
    if args.dataset_version:
        job_config['dataset_version'] = args.dataset_version
    if args.model_type:
        job_config['model_type'] = args.model_type
    if args.job_name:
        job_config['job_name'] = args.job_name
    if args.machine_type:
        job_config['machine_type'] = args.machine_type
    if args.accelerator:
        job_config['accelerator_type'] = args.accelerator
    if args.accelerator_count is not None:
        job_config['accelerator_count'] = args.accelerator_count
    if args.gcs_bucket:
        GCS_BUCKET = args.gcs_bucket
    
    # Submit job with resolved config
    job = submit_training_job(
        job_name=job_config.get('job_name'),
        machine_type=job_config['machine_type'],
        accelerator_type=job_config.get('accelerator_type'),
        accelerator_count=job_config.get('accelerator_count', 0),
        dataset_version=job_config.get('dataset_version'),
        model_type=job_config['model_type'],
    )
    
    # Monitor job if --wait flag is set
    if args.wait:
        try:
            job.wait()
        except KeyboardInterrupt:
    else:
    
    # Print final status (handle case where job resource isn't available yet)
    try:
    except (RuntimeError, AttributeError) as e:
        # Check if it's a quota error
        if "quota" in str(e).lower():
        else:
