import json
from typing import List, Tuple
from sqlalchemy.orm import Session, aliased
from sqlalchemy import and_, func, or_
from uuid import uuid4
from schemas import DatasetVersionInfo, JobListItem, ProcessRegistryModel, FileVersionPair
from exceptions import NoContentError, ConflictError, AIServerError
from entities import Job, JobFiles, Process, JobStatus, DatasetVersion, Dataset

def get_job_list(db: Session):
    """
    Retrieves a list of jobs from the database with their details.

    This function performs a query on the Job table to retrieve various 
    details about each job, including its name, creation timestamp, 
    status, parent job name, and virtual environment name. It uses 
    aliases for the JobStatus and Job tables to join and gather 
    additional information about the job status and parent job. 

    Args:
        db (Session): The SQLAlchemy session used to perform the database query.

    Returns:
        List[JobListItem]: A list of JobListItem objects, each containing 
        the job name, creation timestamp, status name, parent job name, 
        and virtual environment name.
    """

    job_status_alias = aliased(JobStatus)
    parent_job_alias = aliased(Job)
    query = db.query(
        Job.job_name,
        Job.created_timestamp,
        job_status_alias.status_name.label('status_name'),
        parent_job_alias.job_name.label('parent_name'),
        Job.venv_name,
    ).join(
        job_status_alias,
        Job.status_id == job_status_alias.id
    ).outerjoin(
        parent_job_alias,
        Job.parent_job_id == parent_job_alias.id
    ).all()
    return [JobListItem(job_name=row[0], created_timestamp=row[1], status=row[2], parent_job_name=row[3], venv_name=row[4]) for row in query]

def get_base_job_list(db:Session):
    """
    Retrieves a list of base jobs from the database with their details.

    This function performs a query on the Job table to retrieve the name, 
    creation timestamp, and virtual environment name for all base jobs. 
    A base job is a job with no parent job. The function returns a list of 
    JobListItem objects, each containing the job name, creation timestamp, 
    status name (which is None for base jobs), parent job name (which is None 
    for base jobs), and virtual environment name.

    Args:
        db (Session): The SQLAlchemy session used to perform the database query.

    Returns:
        List[JobListItem]: A list of JobListItem objects, each containing the job 
        name, creation timestamp, status name, parent job name, and virtual 
        environment name.
    """
    query = db.query(
        Job.job_name,
        Job.created_timestamp,
        Job.venv_name,
    ).filter(Job.parent_job_id == None).all()
    return [JobListItem(job_name=row[0], created_timestamp=row[1], status=None, parent_job_name=None, venv_name=row[2]) for row in query]



def get_job_by_name(db: Session, job_name: str):
    return db.query(Job).filter(Job.job_name == job_name).first()


def get_processes_of_job_by_job_name(db:Session, job_name: str, only_running_process: bool = False):
    """
    Retrieves a list of processes from the database for a given job name.

    This function performs a query on the Process table to retrieve all processes
    for a given job name. If the only_running_process parameter is set to True,
    the function only returns processes that are currently running (i.e. processes
    where the exit code is None). The function returns a list of Process objects.

    Args:
        db (Session): The SQLAlchemy session used to perform the database query.
        job_name (str): The name of the job for which to retrieve processes.
        only_running_process (bool): Whether to only return running processes.
            Defaults to False.

    Returns:
        List[Process]: A list of Process objects, each containing details of a
            process for the given job.

    Raises:
        NoContentError: If the job named job_name does not exist.
    """
    job_obj = get_job_by_name(db, job_name)
    if not job_obj:
        raise NoContentError(f'The job named "{job_name}" was not found.')
    if only_running_process:
        return db.query(Process).filter(and_(Process.job_id == job_obj.id, Process.exit_code == None)).all()
    return db.query(Process).filter(Process.job_id == job_obj.id).all()

def get_file_and_versions(db: Session, files: List[FileVersionPair]) -> List[DatasetVersionInfo]:
    """
    Retrieves the id and filepath for a list of dataset versions.

    This function takes a list of FileVersionPair objects, each containing a filename and a version number.
    It then performs a query on the DatasetVersion table to retrieve the id and filepath for the specified
    versions of the specified files. If the version number is None for a file, it retrieves the id and filepath
    for the latest version of the file. The function returns a list of DatasetVersionInfo objects, each containing
    the id, source filename, and target filename for a dataset version.

    Args:
        db (Session): The SQLAlchemy session used to perform the database query.
        files (List[FileVersionPair]): A list of FileVersionPair objects, each containing a filename and a version number.

    Returns:
        List[DatasetVersionInfo]: A list of DatasetVersionInfo objects, each containing the id, source filename, and target filename for a dataset version.
    """
    file_objs = []
    for file in files:
        dataset_alias = aliased(Dataset)

        if file.version is None:
            max_version_subq = db.query(
                func.max(DatasetVersion.version_number).label("max_version")
            ).join(
                dataset_alias,
                DatasetVersion.dataset_id == dataset_alias.id
            ).filter(
                dataset_alias.filename == file.filename
            ).group_by(dataset_alias.filename).subquery()

            query = db.query(DatasetVersion.id, DatasetVersion.filepath).join(
                Dataset
            ).filter(
                Dataset.filename == file.filename,
                DatasetVersion.version_number == max_version_subq.c.max_version
            )
        else:
            query = db.query(DatasetVersion.id, DatasetVersion.filepath).join(
                Dataset
            ).filter(
                Dataset.filename == file.filename,
                DatasetVersion.version_number == file.version
            )

        file_objs.extend([DatasetVersionInfo(dataset_version_id=row[0], source_filename=row[1], target_filename=file.filename) for row in query.all()])
    return file_objs


def create_job(db: Session, files: List[FileVersionPair], parent_job_name: str, job_name:str=None, base_job_name: str = None):
    """
    Creates a new job in the database.

    Args:
        db (Session): The SQLAlchemy session used to perform the database query.
        files (List[FileVersionPair]): A list of FileVersionPair objects, each containing a filename and a version number.
        parent_job_name (str): The name of the parent job.
        job_name (str): The name of the job. If not provided, a UUID will be generated.
        base_job_name (str): The name of the base job. If not provided, the parent job's venv_name will be used.

    Returns:
        Tuple[Job, List[DatasetVersionInfo]]: A tuple containing the newly created Job object and a list of DatasetVersionInfo objects.
    """
    parent_job = get_job_by_name(db, parent_job_name) if parent_job_name else None
    base_job = get_job_by_name(db, base_job_name) if base_job_name else None
    if not parent_job_name and parent_job:
        raise NoContentError(f'Parent job named "{parent_job_name}" does not exist!')
    if not base_job_name and base_job:
        raise NoContentError(f'Base job named "{base_job_name}" does not exist!')
    job = Job(
        job_name = job_name if job_name else uuid4(),
        parent_job_id = parent_job.id if parent_job else None,
        base_job_id = base_job.id if base_job else None,
        status_id = 0,
        venv_name = base_job.venv_name if base_job else parent_job.venv_name
    )
    db.add(job)
    db.commit()
    files_info = get_file_and_versions(db, files)
    db.add_all([JobFiles(job_id= job.id, dataset_version_id= file_info.dataset_version_id) for file_info in files_info])
    db.commit()
    return job, files_info

def record_base_job(db: Session, base_job_name: str, venv_name: str):
    """
    Records a base job in the database.

    Args:
        db (Session): The SQLAlchemy session used to perform the database query.
        base_job_name (str): The name of the base job. If not provided, a UUID will be generated.
        venv_name (str): The name of the virtual environment associated with the base job.

    Returns:
        Job: The newly created Job object.

    Raises:
        ConflictError: If a job with the same name already exists.
    """
    job_obj = get_job_by_name(db, base_job_name)
    if job_obj:
        raise ConflictError(f'There is already a job named "{base_job_name}".')
    job = Job(
        job_name = base_job_name if base_job_name else uuid4(),
        parent_job_id = None,
        base_job_id = None,
        status_id = None,
        venv_name = venv_name
    )
    db.add(job)
    db.commit()
    return job

def soft_delete_job(db: Session, job_name: str):
    """
    Soft deletes a job.

    Args:
        db (Session): The SQLAlchemy session used to perform the database query.
        job_name (str): The name of the job to be soft deleted.

    Raises:
        AIServerError: If there are running processes associated with the job.
    """
    job_id = db.query(Job).filter(Job.job_name == job_name).first().id
    process_list = db.query(Process).filter(and_(Process.job_id == job_id, Process.exit_code != None)).all()
    if len(process_list) > 0:
        raise AIServerError("Jobs cannot be deleted while there are running processes!")
    db.query(Job).filter(Job.job_name == job_name).update({"is_deleted": True})
    db.commit()

def run_process(db: Session, pid: int, job_name:str, args: dict) -> int:
    """
    Creates a new process entry in the database for a given job name.

    This function retrieves the job object by its name, creates a new 
    process entry with the given process ID and arguments, and stores 
    it in the database. The function returns the ID of the newly created 
    process.

    Args:
        db (Session): The SQLAlchemy session used to perform the database query.
        pid (int): The process ID.
        job_name (str): The name of the job associated with the process.
        args (dict): Additional arguments for the process, passed as a dictionary.

    Returns:
        int: The ID of the newly created process.
    """

    job_obj = get_job_by_name(db, job_name)
    process_obj = Process(
        pid=pid,
        job_id = job_obj.id,
        args = json.dumps(args)
        )
    db.add(process_obj)
    db.commit()
    return process_obj.id


def process_exit(db:Session, process: ProcessRegistryModel, exit_code:int, end_date):
    """
    Updates a process entry in the database with its exit code and end date.

    This function takes a process registry model object, an exit code, and an end date, 
    and updates the corresponding process entry in the database with the given values.
    The function commits the changes to the database.

    Args:
        db (Session): The SQLAlchemy session used to perform the database query.
        process (ProcessRegistryModel): The process registry model object to be updated.
        exit_code (int): The exit code of the process.
        end_date (datetime): The end date of the process.
    """
    db.query(Process).filter(Process.id == process.db_id).update({"exit_code": exit_code, "end_date": end_date})
    db.commit()