
from datetime import datetime
from pydantic import BaseModel, Field, validator
from uuid import uuid4
from typing import List, TypeVar, Generic, Any, Dict, Union
from pydantic.generics import GenericModel
from typing import TypeVar
from fastapi import Response
import json


T = TypeVar("T", str, int, List, Dict)

class GenericField(GenericModel, Generic[T]):
    data: T = Field(default=None)

class Response(GenericField[T], Generic[T], Response):
    message: str = Field(default="")
    is_success: bool = Field(default=True)

class Generic(BaseModel): # TODO Tüm schemalar bundan generic olacak.
    is_success: bool
    detail: str
    data: BaseModel | None

class FileModel(BaseModel):
    filename: str
    tags: List[str]
    last_version: int


class FileVersionPair(BaseModel):
    filename: str
    version: int | None = None


class CreateJobArgs(BaseModel):
    args: dict = Field(default_factory=dict)
    job_name: str = Field(default_factory=uuid4, max_length=255)
    parent_job_name: Union[str,None] = Field(max_length=255)
    base_job_name: Union[str,None] = Field(max_length=255)
    filenames_and_version:str = Field(default="") # dataset.csv:1,dataset.csv:2

    def parse_filenames_and_version(self) -> List[FileVersionPair]:
        """
        Parses the filenames and versions from a comma-separated string.

        Splits `self.filenames_and_version` by commas to extract each filename and optional version.
        Each entry is further split by a colon. If a version is present, it is parsed as an integer.
        Returns a list of `FileVersionPair` objects with the extracted filenames and versions.

        Returns:
            List[FileVersionPair]: A list of FileVersionPair instances, each containing a filename
            and an optional version.
        """

        tokens = [token.strip() for token in self.filenames_and_version.split(',')]
        result = []
        for token in tokens:
            subtokens = token.split(':')
            if len(subtokens) == 1:
                result.append(FileVersionPair(filename=subtokens[0].strip()))
            else:
                result.append(FileVersionPair(filename=subtokens[0].strip(), version=int(subtokens[1].strip())))
        return result

class ProcessListItem(BaseModel):
    id: int
    pid: int
    args: List[str]
    exit_code: int | None
    start_date: datetime
    end_date: datetime | None


class DatasetVersionInfo(BaseModel):
    dataset_version_id: int
    source_filename: str
    target_filename: str
    
class JobListItem(BaseModel):
    job_name: str
    parent_job_name: str | None
    status: str | None
    created_timestamp: datetime
    venv_name: str


class JobDetail(BaseModel):
    job_name: str
    parent_job_name: str | None
    status: str | None
    created_timestamp: datetime
    venv_name: str
    files: List[FileVersionPair]
    job_args: List[dict]
    processes: List[ProcessListItem]



class RunJobArgs(BaseModel):
    args: List[str] = Field(default_factory=list)

class JobArgsFileModel(BaseModel):
    job_name: str
    parent_job_name: str | None
    base_job_name: str | None
    args: dict

    def to_dict(self):
        return {"job_name": self.job_name, "base_job_name": self.base_job_name, "parent_job_name": self.parent_job_name, "args": self.args}

class ProcessRegistryModel(BaseModel):
    pid: int
    db_id: int