
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

from pydantic import ValidationError
from schemas import CreateJobArgs, DatasetVersionInfo, JobArgsFileModel, ProcessRegistryModel
import settings
import json
import os
import shutil
from exceptions import ConflictError, NoContentError, FileSystemError, NotValidError
import psutil
import subprocess
from fine_tune_ops.process_management import ProcessScanner 
import sys
import glob

def copy_folder(src_folder_path: str | Path, dst_folder_path: str | Path, ignore_pattern: List[str]):
    """
    Copies all files and directories from src_folder_path to dst_folder_path,
    except for the files and directories that match the patterns in ignore_pattern.
    """
    items = [item for item in glob.glob(os.path.join(src_folder_path, '*'))
            if not any(glob.fnmatch.fnmatch(item, ignore) for ignore in ignore_pattern)]
    for item in items:
        if os.path.isfile(item):
            shutil.copy(item, os.path.join(dst_folder_path, os.path.basename(item)))
        elif os.path.isdir(item):
            shutil.copytree(item, os.path.join(dst_folder_path, os.path.basename(item)))

class JobIO:

    def __init__(self, job_name: str) -> None:
        self.job_name = job_name
    
    def exist_job(self) -> bool:
        """
        Checks if the job with the given name exists.

        Returns:
            bool: True if the job exists, False otherwise.
        """
        return JobIO.exist_job_by_name(self.job_name)
    
    
    def is_valid(self, args: dict, base_job_name: str) -> Tuple[bool, ValidationError]:
        """
        Check if the given job arguments are valid according to the validation module
        specified in settings.JOB_VALIDATE_MODULE.

        Args:
            args (dict): The job arguments to validate.
            base_job_name (str): The name of the base job.

        Returns:
            Tuple[bool, ValidationError]: A tuple where the first element is a boolean indicating
            whether the arguments are valid, and the second element is a ValidationError exception
            if the arguments are not valid.
        """
        base_job_io = JobIO(base_job_name)
        sys.path.append(str(base_job_io.get_job_dir_path()))
        validate = __import__(settings.JOB_VALIDATE_MODULE)
        try:
            validate.ArgsModel(**args)
            result = True, None
        except ValidationError as e:
            result = False, e
        del validate
        sys.path.pop()
        return result
    
    @staticmethod
    def exist_job_by_name(job_name: str) -> bool:
        """
        Checks if the job with the given name exists.

        Args:
            job_name (str): The name of the job to check.

        Returns:
            bool: True if the job exists, False otherwise.
        """
        job_path = settings.ROOT_DIR_PATH / settings.JOBS_DIRNAME / job_name
        return os.path.isdir(job_path)
        
    def get_job_dir_path(self) -> Path:
        return settings.ROOT_DIR_PATH / settings.JOBS_DIRNAME / self.job_name
    
    def create_job_folders(self):
        os.makedirs(self.get_job_dir_path() / settings.JOB_OUTPUT_DIRNAME, exist_ok=True)
        os.makedirs(self.get_job_dir_path() / settings.JOB_LOGS_DIRNAME, exist_ok=True)
        os.makedirs(self.get_job_dir_path() / settings.JOB_FILES_DIRNAME, exist_ok=True)
    
    def copy_from_parent_jobs(self, parent_job_name: str):
        """
        Copies all files and directories from the parent job's directory to this job's directory,
        except the job_args.json file.
        
        Args:
            parent_job_name (str): The name of the parent job.
        """
        src_folder_path = settings.ROOT_DIR_PATH / settings.JOBS_DIRNAME / parent_job_name
        dst_folder_path = self.get_job_dir_path()
        items = [settings.JOB_LOGS_DIRNAME, settings.JOB_OUTPUT_DIRNAME, settings.JOB_ARGS_FILENAME, settings.JOB_FILES_DIRNAME]
        for item in items:
            if os.path.isdir(src_folder_path / item):
                if os.path.exists(self.get_job_dir_path() / item):
                    shutil.rmtree(self.get_job_dir_path() / item)
                shutil.copytree(src_folder_path / item, self.get_job_dir_path() / item)
            elif os.path.isfile(src_folder_path / item):
                if os.path.exists(self.get_job_dir_path() / item):
                    os.remove(self.get_job_dir_path() / item)
                shutil.copyfile(src_folder_path / item, self.get_job_dir_path() / item)

    def copy_from_base_jobs(self, base_job_name: str):
        """
        Copies all files and directories from the base job's directory to the current job's directory,
        excluding specific directories and files defined in the ignore pattern.

        Args:
            base_job_name (str): The name of the base job whose files and directories are to be copied.
        """

        src_folder_path = settings.ROOT_DIR_PATH / settings.JOBS_DIRNAME / base_job_name
        dst_folder_path = self.get_job_dir_path()
        ignore_pattern = [settings.JOB_LOGS_DIRNAME, settings.JOB_OUTPUT_DIRNAME, settings.JOB_ARGS_FILENAME, settings.JOB_FILES_DIRNAME]
        copy_folder(src_folder_path, dst_folder_path, ignore_pattern)

    
    def create_job_args_file(self, parent_job_name:str, base_job_name:str, args: dict):
        """
        Creates or updates the job arguments file for the current job.

        This function reads the existing job arguments from a JSON file, 
        creates a new job argument entry with the provided parameters, 
        and writes the updated list back to the file.

        Args:
            parent_job_name (str): The name of the parent job.
            base_job_name (str): The name of the base job.
            args (dict): A dictionary containing the arguments for the job.
        """

        args_path = self.get_job_dir_path() / settings.JOB_ARGS_FILENAME
        if os.path.exists(args_path):
            with open(args_path, 'r', encoding="utf-8") as fs:
                data = json.load(fs)
                try:
                    base_args = [JobArgsFileModel(**row).model_dump() for row in data]
                except:
                    base_args = [JobArgsFileModel(**row).dict() for row in data]
        else:
            base_args = []
        with open(args_path, 'w', encoding="utf-8") as fs:
            item = JobArgsFileModel(job_name=self.job_name, base_job_name=base_job_name, parent_job_name=parent_job_name, args=args)
            
            base_args.insert(0, item.dict())
            json.dump(base_args, fs, indent=4)
    
    def create_utils_file(self, parent_job_name: str, base_job_name: str):
        """
        Creates a server_utils.py file in the job directory with the given job name,
        parent job name and base job name. The file is a Jinja2 template and is rendered
        with the given data.

        Args:
            parent_job_name (str): The name of the parent job.
            base_job_name (str): The name of the base job.
        """
        from jinja2 import Environment, FileSystemLoader
        file_loader = FileSystemLoader(settings.STATIC_FILES_PATH)
        env = Environment(loader=file_loader)
        template = env.get_template("server_utils.py.jinja")
        data = {
            'settings': settings.__dict__,
            'job_name': self.job_name,
            'parent_job_name': parent_job_name,
            'base_job_name': base_job_name
        }
        rendered_template = template.render(data)
        with open(self.get_job_dir_path() / 'server_utils.py', 'w') as file:
            file.write(rendered_template)

    
    def create_job(self, req_args: CreateJobArgs):
        """
        Creates a job with the given arguments.

        If the job folder already exists, this method raises a ConflictError.
        If there is an error during the creation of the job, the job folder is deleted
        and the exception is re-raised.

        This method copies all files and directories from the parent job and the base job
        to the current job, except the job_args.json file.
        Then it creates a job arguments file with the given parameters.
        Finally, it creates a server_utils.py file in the job directory.

        Args:
            req_args (CreateJobArgs): The arguments for the job to be created.
        """
        if self.exist_job():
            raise ConflictError(f'There is already a folder named "{req_args.job_name}"!')
        try:
            os.makedirs(self.get_job_dir_path(), exist_ok=True)
            if req_args.base_job_name is not None:
                self.copy_from_base_jobs(req_args.base_job_name)
            if req_args.parent_job_name is not None:
                self.copy_from_parent_jobs(req_args.parent_job_name)
            self.create_job_folders()
            self.create_job_args_file(req_args.parent_job_name, req_args.base_job_name, req_args.args)
            self.create_utils_file(req_args.parent_job_name, req_args.base_job_name)
        except Exception as e:
            self.hard_delete_job()
            raise e
    
    def files_copy_to_job(self, files_info: List[DatasetVersionInfo]):
        """
        Copies files from the source path to the target path.

        The source path is the filename in the FILES_DIRNAME directory.
        The target path is the filename in the JOB_FILES_DIRNAME directory in the job directory.

        Args:
            files_info (List[DatasetVersionInfo]): A list of DatasetVersionInfo objects.
        """
        for file_info in files_info:
            src_path = settings.ROOT_DIR_PATH /  settings.FILES_DIRNAME / file_info.source_filename
            tar_path = self.get_job_dir_path() / settings.JOB_FILES_DIRNAME / file_info.target_filename
            shutil.copyfile(src_path, tar_path)


    def hard_delete_job(self):
        """
        Permanently deletes the job directory if it exists.

        This method checks if a job directory with the current job name exists.
        If it does, it raises a ConflictError to indicate that a folder with
        the same name already exists. If the directory is found, it removes
        the directory from the filesystem.

        Raises:
            ConflictError: If a job with the same name already exists.
        """

        if self.exist_job():
            raise ConflictError(f'There is already a folder named "{self.job_name}"!')
        if os.path.isdir(self.get_job_dir_path()):
            os.rmdir(self.get_job_dir_path())
    
    def run(self, process_args: List[str], venv_name: str):
        """
        Runs a process in the current job directory.

        Args:
            process_args (List[str]): A list of arguments for the process.
            venv_name (str): The name of the virtual environment to use.

        Returns:
            subprocess.Popen: The process object.
        """
        if os.name == 'nt': venv_path = settings.ROOT_DIR_PATH / settings.VENVS_DIRNAME / venv_name / "Scripts" / "activate"
        elif os.name == 'posix': venv_path = settings.ROOT_DIR_PATH / settings.VENVS_DIRNAME / venv_name / "bin" / "python"
        job_path = settings.ROOT_DIR_PATH / settings.JOBS_DIRNAME / self.job_name
        output_path = os.path.join(str(job_path) ,settings.JOB_LOGS_DIRNAME ,f"output_{datetime.now().isoformat().replace(':', '.')}.log")
        arg_str = '' if len(process_args) == 0 else '"' +'" "'.join(process_args) + '"'
        command = f'cd "{job_path}" && "{venv_path}" main.py {arg_str} > "{output_path}" '
        # with open(job_path / settings.JOB_LOGS_DIRNAME / f"output_{datetime.now().isoformat().replace(':', '.')}.log", "w") as fs_out:
        #     with open(job_path / settings.JOB_LOGS_DIRNAME / f"error_{datetime.now().isoformat().replace(':', '.')}.log", "w") as fs_err:
        process = subprocess.Popen(command, shell=True,cwd= str(job_path))
        return process
        



class ProcessIO:

    def __init__(self, pid:int) -> None:
        """
        Constructor for ProcessIO.
        
        Args:
            pid (int): The process id of the process to monitor.
        
        Attributes:
            pid (int): The process id of the process to monitor.
            process (psutil.Process): The process object.
        """
        self.pid = pid
        self.process = psutil.Process(self.pid)

    @staticmethod
    def is_exists(pid:int) -> bool:
        return psutil.pid_exists(pid)
    
    def kill(self):
        self.process.kill()
    
    def status(self):
        return self.process.status()
    
    def memory_info(self):
        return self.process.memory_info()
    
    def name(self):
        return self.process.name()

    def cwd(self):
        return self.process.cwd()
    
    def cmdline(self):
        return self.process.cmdline()
    
