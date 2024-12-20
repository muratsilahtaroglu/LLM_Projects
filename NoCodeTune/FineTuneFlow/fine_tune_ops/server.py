from fastapi import Depends, FastAPI,status
from fine_tune_ops.process_management import ProcessScanner
from logs import Logger
from typing import List
import settings
from entities import get_db
from schemas import *
from sqlalchemy.orm import Session
import fine_tune_ops.crud as crud
import fine_tune_ops.crud_io as crud_io
import exceptions

app = FastAPI()
exceptions.config_exception_handler(app)


@app.get("/jobs", response_model=List[JobListItem], status_code=status.HTTP_200_OK)
async def job_list(db: Session = Depends(get_db)):
    """
    Get a list of jobs from the database with their details.

    This endpoint returns a list of JobListItem objects, each containing 
    the job name, creation timestamp, status name, parent job name, 
    and virtual environment name.

    Args:
        db (Session): The SQLAlchemy session used to perform the database query.

    Returns:
        List[JobListItem]: A list of JobListItem objects, each containing 
        the job name, creation timestamp, status name, parent job name, 
        and virtual environment name.
    """
    jobs = crud.get_job_list(db)
    return [job for job in jobs]

@app.get("/jobs/bases", response_model=List[JobListItem], status_code=status.HTTP_200_OK)
async def job_base_list(db: Session = Depends(get_db)):
    """
    Get a list of base jobs from the database with their details.

    This endpoint returns a list of JobListItem objects, each containing 
    the job name, creation timestamp, and virtual environment name. 
    Base jobs are jobs with no parent job.

    Args:
        db (Session): The SQLAlchemy session used to perform the database query.

    Returns:
        List[JobListItem]: A list of JobListItem objects, each containing 
        the job name, creation timestamp, and virtual environment name.
    """

    jobs = crud.get_base_job_list(db)
    return [job for job in jobs]


@app.post("/jobs/build", response_model=bool, status_code=status.HTTP_201_CREATED)
async def build_job(job_args:CreateJobArgs, db: Session = Depends(get_db)):
    """
    Create a new job, given a set of arguments.

    Args:
    - job_args: A CreateJobArgs object, containing the arguments to create a new job.

    Returns:
    - A boolean indicating whether the job was successfully created.

    Raises:
    - exceptions.NotValidError: If the base_job and parent_job are both None.
    - exceptions.FileSystemError: If the parent job or base job folder does not exist.
    - exceptions.ConflictError: If a job already exists with the given name.
    - exceptions.BaseJobError: If the base job specified is not a valid base job.
    """
    if job_args.parent_job_name is None and job_args.base_job_name is None:
        raise exceptions.NotValidError("base_job and parent_job cannot be both None.")
    if job_args.parent_job_name and not crud_io.JobIO.exist_job_by_name(job_args.parent_job_name):
        raise exceptions.FileSystemError(f'Parent job folder named "{job_args.parent_job_name}" could not be found.')
    if job_args.base_job_name and not crud_io.JobIO.exist_job_by_name(job_args.base_job_name):
        raise exceptions.FileSystemError(f'Base job folder named "{job_args.base_job_name}" could not be found.')
    job_obj = crud.get_job_by_name(db, job_args.job_name)
    job_io = crud_io.JobIO(job_args.job_name)
    if job_obj:
        raise exceptions.ConflictError("There is already a job with this name.")
    is_valid, ex = job_io.is_valid(job_args.args, job_args.base_job_name)
    if not is_valid:
        raise ex
    job_io.create_job(job_args)
    job, file_info = crud.create_job(
        db,
        files=job_args.parse_filenames_and_version(),
        parent_job_name=job_args.parent_job_name,
        job_name= job_args.job_name,
        base_job_name=job_args.base_job_name
    )
    job_io.files_copy_to_job(file_info)
    return True

@app.post("/jobs/{job_name}/run", response_model=int, status_code=status.HTTP_202_ACCEPTED)
async def run_job(job_name:str, job_args:RunJobArgs, db: Session = Depends(get_db)):
    """
    Runs a job with the specified name and arguments.

    This function retrieves a job by its name from the database and runs it using
    the provided arguments. It updates the process registry with the new process
    information and returns the process ID.

    Args:
    - job_name (str): The name of the job to run.
    - job_args (RunJobArgs): The arguments required to run the job.
    - db (Session): The database session dependency.

    Returns:
    - int: The process ID of the started job.

    Raises:
    - exceptions.NoContentError: If the job is not found in the database.
    """

    job_obj =  crud.get_job_by_name(db, job_name)
    if not job_obj:
        raise exceptions.NoContentError("Job not found!")
    job_io = crud_io.JobIO(job_name)
    process = job_io.run(job_args.args, job_obj.venv_name)
    process_db_id = crud.run_process(db, pid=process.pid, job_name=job_name, args=job_args.args)
    ProcessScanner().add_process(ProcessRegistryModel(pid=process.pid, db_id=process_db_id))
    return process.pid

@app.get("/jobs/{job_name}/process", response_model=List[ProcessListItem], status_code=status.HTTP_200_OK)
async def run_job(job_name:str, db: Session = Depends(get_db)):
    """
    Retrieves a list of active processes for a job with the given name.

    This function retrieves a job by its name from the database and retrieves all active
    processes for that job. It returns a list of ProcessListItem objects containing the
    process ID, arguments, exit code, start date, and end date for each process.

    Args:
    - job_name (str): The name of the job to retrieve processes for.
    - db (Session): The database session dependency.

    Returns:
    - List[ProcessListItem]: A list of ProcessListItem objects.

    Raises:
    - exceptions.NoContentError: If the job is not found in the database.
    """
    job_obj =  crud.get_job_by_name(db, job_name)
    if not job_obj:
        raise exceptions.NoContentError("Job not found!")
    process_objs = crud.get_processes_of_job_by_job_name(db, job_name, False)
    return [ProcessListItem(id=obj.id, args=json.loads(obj.args), pid=obj.pid, exit_code= obj.exit_code, start_date=obj.start_date, end_date=obj.end_date) for obj in process_objs]

@app.get("/process/{pid}/stop", response_model=bool, status_code=status.HTTP_202_ACCEPTED)
async def run_job(pid:int, db: Session = Depends(get_db)):
    """
    Stops a process with the given PID.

    This function stops a process with the given PID and returns True if successful.

    Args:
    - pid (int): The PID of the process to stop.
    - db (Session): The database session dependency.

    Returns:
    - bool: True if the process was successfully stopped.

    Raises:
    - exceptions.FileSystemError: If the process doesn't exist.
    """
    if not crud_io.ProcessIO.is_exists(pid):
        raise exceptions.FileSystemError("Process doesn't exist anyway!")
    process_obj = crud_io.ProcessIO(pid)
    process_obj.kill()
    return True

@app.delete("/jobs/{job_name}", response_model=bool, status_code=status.HTTP_202_ACCEPTED)
async def delete_job(job_name:str, db: Session = Depends(get_db)):
    """
    Soft deletes a job from the database.

    This function sets the is_deleted flag of a job to True in the database,
    effectively soft deleting the job.

    Args:
    - job_name (str): The name of the job to soft delete.
    - db (Session): The database session dependency.

    Returns:
    - bool: True if the job was successfully soft deleted.

    Raises:
    - exceptions.NoContentError: If the job is not found in the database.
    """
    crud.soft_delete_job(db, job_name)
    return True



@app.post("/admin/base_job/record", response_model=bool, status_code=status.HTTP_201_CREATED)
async def record_base_job(job_name:str, venv_name:str, db: Session = Depends(get_db)):
    """
    Records a base job with the specified job name and virtual environment name.

    This function checks if a job with the given name exists. If it does, the base job is recorded
    in the database with the provided virtual environment name. If the job does not exist, a
    NoContentError is raised.

    Args:
    - job_name (str): The name of the job to record.
    - venv_name (str): The name of the virtual environment associated with the job.
    - db (Session): The database session dependency.

    Returns:
    - bool: True if the base job was successfully recorded.

    Raises:
    - exceptions.NoContentError: If the base job folder is not found.
    """

    job_io = crud_io.JobIO(job_name)
    if job_io.exist_job():
        crud.record_base_job(db, base_job_name=job_name, venv_name=venv_name)
    else:
        raise exceptions.NoContentError("Base job folder not found!")
    return True

@app.post("/venvs", response_model=bool, status_code=status.HTTP_201_CREATED)
async def venv_create(venv_name:str, packets: List[str], db: Session = Depends(get_db)):
    #TODO: will be implemented later
    raise NotImplementedError()

@app.get("/venvs", response_model=List[str], status_code=status.HTTP_201_CREATED)
async def venvs_list(db: Session = Depends(get_db)):
     #TODO: will be implemented later
    raise NotImplementedError()

@app.get("/venvs/{venv_name}", response_model=List[str], status_code=status.HTTP_201_CREATED)
async def venv_detail(venv_name:str, db: Session = Depends(get_db)):
     #TODO: will be implemented later
    raise NotImplementedError()

@app.post("/venvs/{venv_name}", response_model=bool, status_code=status.HTTP_201_CREATED)
async def venv_packet_install(venv_name:str, packets: List[str], db: Session = Depends(get_db)):
     #TODO: will be implemented later
    raise NotImplementedError()

@app.delete("/venvs/{venv_name}", response_model=bool, status_code=status.HTTP_201_CREATED)
async def venv_delete(venv_name:str, db: Session = Depends(get_db)):
     #TODO: will be implemented later
    raise NotImplementedError()