from typing import List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, or_
from uuid import uuid4
from schemas import FileModel

from entities import Dataset, DatasetVersion

def get_dataset_list(db: Session) -> List[FileModel]:
    """
    Return a list of datasets with their last version number

    Args:
        db: The database session to use

    Returns:
        A list of FileModel objects, each one representing a dataset with its last version number
    """
    query = db.query(Dataset.filename, Dataset.tags, func.max(DatasetVersion.version_number)) \
        .join(DatasetVersion, Dataset.id == DatasetVersion.dataset_id) \
        .filter(DatasetVersion.is_deleted == False) \
        .group_by(Dataset.filename, Dataset.tags) \
        .all()
    return [FileModel(filename=row[0], tags=row[1].split(','), last_version=row[2]) for row in query]
    

def get_dataset_by_filename(db: Session, filename: str):
    """
    Return the dataset with the given filename

    Args:
        db: The database session to use
        filename: The filename of the dataset to retrieve

    Returns:
        The dataset with the given filename, or None if not found
    """
    return db.query(Dataset).filter(Dataset.filename == filename).first()

def get_dataset_versions_by_filename(db: Session, filename:str, import_deleted:bool = False):
    """
    Return all the versions of the dataset with the given filename

    Args:
        db: The database session to use
        filename: The filename of the dataset to retrieve
        import_deleted: If True, also return deleted versions

    Returns:
        A list of DatasetVersion objects, or None if no dataset with the given filename is found
    """
    dataset = get_dataset_by_filename(db, filename)
    if not dataset: return None
    if import_deleted:
        return db.query(DatasetVersion).filter(and_(DatasetVersion.dataset_id == dataset.id)).all()
    return db.query(DatasetVersion).filter(and_(DatasetVersion.dataset_id == dataset.id, DatasetVersion.is_deleted == False)).all()


def get_dataset_versions_by_dataset_id(db: Session, dataset_id:str, import_deleted:bool = False):
    """
    Return all the versions of the dataset with the given id

    Args:
        db: The database session to use
        dataset_id: The id of the dataset to retrieve
        import_deleted: If True, also return deleted versions

    Returns:
        A list of DatasetVersion objects, or None if no dataset with the given id is found
    """
    
    if import_deleted:
        return db.query(DatasetVersion).filter(and_(DatasetVersion.dataset_id == dataset_id)).all()
    return db.query(DatasetVersion).filter(and_(DatasetVersion.dataset_id == dataset_id, DatasetVersion.is_deleted == False)).all()

def get_avaible_dataset(db: Session, filenames: List[str]) -> List[Dataset]:
    """
    Return the datasets with the given filenames

    Args:
        db: The database session to use
        filenames: The list of filenames of the datasets to retrieve

    Returns:
        A list of Dataset objects, or an empty list if no dataset with the given filenames is found
    """
    return db.query(Dataset).filter(or_(Dataset.filename == filename for filename in filenames)).all()

def generate_new_version_number(db: Session, filename:str = None) -> int:
    """
    Return the next version number for the dataset with the given filename

    Args:
        db: The database session to use
        filename: The filename of the dataset to retrieve the next version number

    Returns:
        The next version number for the dataset with the given filename,
        or None if no dataset with the given filename is found
    """
    dataset = get_dataset_by_filename(db, filename)
    if not dataset: return None
    version = db.query(func.max(DatasetVersion.version_number)).filter(
        and_(
            DatasetVersion.dataset_id == dataset.id
        )
    ).first()
    if not version:
        return 1
    else:
        return version[0] + 1

def create_dataset(db: Session, filename:str, tags:List[str], classification_type:str) -> Tuple[Dataset, DatasetVersion]:
    """
    Create a new dataset with a first version

    Args:
        db: The database session to use
        filename: The filename of the dataset to create
        tags: The list of tags of the dataset to create
        classification_type: The classification type of the dataset to create

    Returns:
        A tuple containing the created dataset and its first version
    """
    
    dataset_obj = Dataset(
            filename = filename,
            tags = ','.join(tags),
            classification_type = classification_type
    )
    db.add(dataset_obj)
    db.commit()
    file_hash = str(uuid4())
    dataset_version_obj = DatasetVersion(
        dataset_id = dataset_obj.id,
        filepath = file_hash,
        version_number = 1
    )
    db.add(dataset_version_obj)
    db.commit()
    return dataset_obj, dataset_version_obj



def create_or_update_dataset(db: Session, filename:str, tags:List[str], classification_type:str) -> Tuple[Dataset, DatasetVersion]:
    """
    Create a new dataset with a first version if it does not exist, or update an existing dataset with a new version if it does exist.

    Args:
        db: The database session to use
        filename: The filename of the dataset to create or update
        tags: The list of tags of the dataset to create or update
        classification_type: The classification type of the dataset to create or update

    Returns:
        A tuple containing the created or updated dataset and its new version
    """
    dataset_obj = get_dataset_by_filename(db, filename)
    if not dataset_obj:
        dataset_obj = Dataset(
                filename = filename,
                tags = ','.join(tags),
                classification_type = classification_type
        )
        db.add(dataset_obj)
        db.commit()
        file_hash = str(uuid4())
        dataset_version_obj = DatasetVersion(
            dataset_id = dataset_obj.id,
            filepath = file_hash,
            version_number = 1
        )
        db.add(dataset_version_obj)
        db.commit()
    else:
        db.query(Dataset).filter(Dataset.id == dataset_obj.id).update({"is_deleted": False, "tags": ','.join(tags), "classification_type": classification_type})
        version_number = generate_new_version_number(db, filename)
        file_hash = str(uuid4())
        dataset_version_obj = DatasetVersion(
            dataset_id = dataset_obj.id,
            filepath = file_hash,
            version_number = version_number
        )
        db.add(dataset_version_obj)
        db.commit()
    return dataset_obj, dataset_version_obj

def delete_dataset_by_filename(db: Session, filename:str):
    """
    Delete the dataset with the given filename and all its versions.

    Args:
        db: The database session to use
        filename: The filename of the dataset to delete
    """
    dataset_obj = get_dataset_by_filename(db, filename)
    db.query(DatasetVersion).filter(
        DatasetVersion.dataset_id == dataset_obj.id
    ).update({"is_deleted": True})
    db.query(Dataset).filter(
        Dataset.id == dataset_obj.id
    ).update({"is_deleted": True})
    db.commit()

def delete_dataset_version_by_filename_and_version_number(db: Session, filename:str, version_number: int):
    """
    Delete the dataset version with the given filename and version number.

    Args:
        db: The database session to use
        filename: The filename of the dataset to delete
        version_number: The version number of the dataset to delete
    """
    dataset_obj = get_dataset_by_filename(db, filename)
    db.query(DatasetVersion).filter(
        and_(
            DatasetVersion.dataset_id == dataset_obj.id,
            DatasetVersion.version_number == version_number
        )
    ).update({"is_deleted": True})
    db.commit()