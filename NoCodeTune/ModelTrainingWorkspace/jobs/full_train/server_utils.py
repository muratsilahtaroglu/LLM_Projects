# This file will be automatically created here by fastapi.
from typing import List
import json


JOB_ARGS_PATH = "job_args.json"

JOB_NAME = "demo_job"
PARENT_JOB_NAME = "None"

FILES_DIRNAME = "files"
OUTPUTS_DIRNAME = "outputs"
LOGS_DIRNAME = "logs"

class JobArgs:
    def __init__(self, job_name: str, parent_job_name: str, args: dict, base_job_name:str):
        self.job_name = job_name
        self.parent_job_name = parent_job_name
        self.args = args
        self.base_job_name = base_job_name

def get_job_args() -> List[JobArgs]:
    with open(JOB_ARGS_PATH, "r", encoding="utf-8") as fs:
        obj = json.load(fs)
        result = [JobArgs(**arg) for arg in obj]
    return result