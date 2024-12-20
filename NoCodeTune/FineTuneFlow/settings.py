from pathlib import Path
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
work_space_folder = "ModelTrainingWorkspace"
ROOT_DIR_PATH = os.path.join(os.path.dirname(current_dir), work_space_folder)
STATIC_FILES_PATH = os.path.join(current_dir,"static_files")

# Static files
UTILS_FILENAME = "server_utils.py"

# Files
FILES_DIRNAME = "files"

# Jobs
JOBS_DIRNAME = "jobs"
JOB_FILES_DIRNAME = "files"
JOB_ARGS_FILENAME = "job_args.json"
JOB_LOGS_DIRNAME = "logs"
JOB_OUTPUT_DIRNAME = "outputs"
JOB_VALIDATE_MODULE = "validate"

# Venv
VENVS_DIRNAME = "venvs"

# Database
DATABASE_CONNECTION_STRING = f"sqlite:///{str(ROOT_DIR_PATH)}\\db-dev.db"

