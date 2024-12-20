from fastapi import FastAPI, HTTPException, UploadFile ,File,Request,status  
from typing import List
from fastapi import Depends
from typing import Annotated
from entities import get_db
from schemas import *
from sqlalchemy.orm import Session
import settings
import data_load.crud as crud

app = FastAPI()


@app.put("/files", response_model=bool, status_code=status.HTTP_201_CREATED)
async def file_upload(
    data_files: Annotated[list[UploadFile], File(description="Multiple files as UploadFile")],
    tags:List[str] = None,
    db: Session = Depends(get_db)
    ):
        """
        Upload a file.

        Args:
        - data_files: A list of files to upload
        - tags: A list of tags to associate with the dataset
        - db: The database session

        Returns:
        - A boolean indicating whether the upload was successful

        Raises:
        - HTTPException: If some of the files already exist, with a 409 status code
        """
        avaible_dataset_versions = crud.get_avaible_dataset(db, [file.filename for file in data_files])
        if len(avaible_dataset_versions) > 0:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Some files you are trying to send already exist!")
        for file in data_files:
            dataset_obj, dataset_version_obj = crud.create_dataset(
                db, 
                filename=file.filename,
                tags=tags,
                classification_type=file.content_type
            )
            filehash = dataset_version_obj.filepath
            with open(settings.ROOT_DIR_PATH / settings.FILES_DIRNAME / filehash, "wb") as fs:
                fs.write(file.file.read())
        
        return True

@app.post("/files", response_model=bool, status_code=status.HTTP_201_CREATED)
async def file_upload(
    data_files: Annotated[list[UploadFile], File(description="Multiple files as UploadFile")],
    tags:List[str] = None,
    db: Session = Depends(get_db)
    ):
        """
        Upload a file, overwriting if it already exists.

        Args:
        - data_files: A list of files to upload
        - tags: A list of tags to associate with the dataset
        - db: The database session

        Returns:
        - A boolean indicating whether the upload was successful
        """
        for file in data_files:
            dataset_obj, dataset_version_obj = crud.create_or_update_dataset(db, filename=file.filename, tags=tags, classification_type=file.content_type)
            with open(settings.ROOT_DIR_PATH / settings.FILES_DIRNAME / dataset_version_obj.filepath, "wb") as fs:
                fs.write(file.file.read())
        
        return True

@app.get("/files", response_model=List[FileModel])
async def file_list(db: Session = Depends(get_db)):
    """
    Retrieve a list of all file datasets.

    Args:
    - db: The database session

    Returns:
    - A list of FileModel instances representing the datasets

    Raises:
    - HTTPException: If an error occurs during database access
    """

    return crud.get_dataset_list(db)

@app.get("/files/{filename}/versions", response_model=List[int])
async def file_list(filename:str, db: Session = Depends(get_db)):
    """
    Retrieve a list of all versions of a file dataset.

    Args:
    - filename: The filename of the dataset to retrieve
    - db: The database session

    Returns:
    - A list of integers representing the versions of the dataset

    Raises:
    - HTTPException: If the dataset does not exist
    """
    dataset_versions = crud.get_dataset_versions_by_filename(db, filename=filename)
    if not dataset_versions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such dataset was found!")
    return [dataset_version.version_number for dataset_version in dataset_versions]

@app.delete("/files/{filename}", response_model=str, status_code=status.HTTP_202_ACCEPTED)
async def delete_file(filename:str, db: Session = Depends(get_db)):
    """
    Delete a file dataset by its filename.

    Args:
    - filename: The name of the dataset to delete
    - db: The database session

    Returns:
    - The filename of the deleted dataset

    Raises:
    - HTTPException: If an error occurs during the deletion process
    """

    crud.delete_dataset_by_filename(db, filename)
    return filename


@app.delete("/files/{filename}/versions/{version}", response_model=dict, status_code=status.HTTP_202_ACCEPTED)
async def delete_file(filename:str, version:int, db: Session = Depends(get_db)):
    """
    Delete a specific version of a file dataset by its filename and version number.

    Args:
    - filename: The name of the dataset to delete
    - version: The version number of the dataset to delete
    - db: The database session

    Returns:
    - A dictionary containing the filename and version of the deleted dataset

    Raises:
    - HTTPException: If an error occurs during the deletion process
    """
    crud.delete_dataset_version_by_filename_and_version_number(db, filename=filename, version_number=version)
    return {"filename": filename, "version": version}
    
