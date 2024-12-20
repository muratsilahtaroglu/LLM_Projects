import settings
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, ARRAY, Boolean, JSON
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import UniqueConstraint

Base = declarative_base()

engine = create_engine(settings.DATABASE_CONNECTION_STRING)

class JobStatus(Base):
    __tablename__ = 'job_status'

    id = Column(Integer, primary_key=True, autoincrement=True)
    status_name = Column(String, nullable=False)
    completed_status = Column(Integer, nullable=True, default=0)

class Process(Base):
    __tablename__ = 'process'
    id = Column(Integer, primary_key=True, autoincrement=True)
    pid = Column(Integer, nullable=False)
    job_id = Column(Integer, ForeignKey('job.id'), nullable=False)
    exit_code = Column(Integer, nullable=True, default=None)
    start_date = Column(DateTime, nullable=True, default=datetime.utcnow)
    end_date = Column(DateTime, nullable=True, default=None)
    args = Column(JSON, nullable=True, default=None)
    job = relationship('Job', foreign_keys=[job_id])


class Job(Base):
    __tablename__ = 'job'

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_name = Column(String(255), nullable=False, unique=True)
    parent_job_id = Column(Integer, ForeignKey('job.id'), nullable=True)
    base_job_id = Column(Integer, ForeignKey('job.id'), nullable=True)
    status_id = Column(Integer, ForeignKey('job_status.id'), nullable=True)
    created_timestamp = Column(DateTime, nullable=True, default=datetime.utcnow)
    venv_name = Column(String(255), nullable=False)
    is_deleted = Column(Boolean, nullable=False, default=False)

    status = relationship('JobStatus', foreign_keys=[status_id])
    parent_job = relationship('Job', foreign_keys=[parent_job_id])
    base_job = relationship('Job', foreign_keys=[base_job_id])


class JobFiles(Base):
    __tablename__ = 'job_files'
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey('job.id'), nullable=False)
    dataset_version_id = Column(Integer, ForeignKey('dataset_version.id'), nullable=False)

    job = relationship('Job', foreign_keys=[job_id])
    dataset_version = relationship('DatasetVersion', foreign_keys=[dataset_version_id])

class Dataset(Base):
    __tablename__ = 'dataset'

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(Text, nullable=False, unique=True)
    classification_type = Column(Text, nullable=False)
    # tags = Column(ARRAY(Text), nullable=True)
    tags = Column(Text, nullable=True)
    created_timestamp = Column(DateTime, nullable=True, default=datetime.utcnow)
    is_deleted = Column(Boolean, nullable=False, default=False)

class DatasetVersion(Base):
    __tablename__ = 'dataset_version'

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey('dataset.id'), nullable=False)
    version_number = Column(Integer, nullable=False)
    filepath = Column(Text, nullable=False)
    created_timestamp = Column(DateTime, nullable=True, default=datetime.utcnow)
    is_deleted = Column(Boolean, nullable=False, default=False)

    dataset = relationship('Dataset', foreign_keys=[dataset_id])
    __table_args__ = (
        UniqueConstraint('dataset_id', 'version_number', name='uq_dataset_id_version_number'),
    )

class Venv(Base):
    __tablename__ = 'venv'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False, unique=True)

class VenvPacket(Base):
    __tablename__ = 'venv_packet'

    id = Column(Integer, primary_key=True, autoincrement=True)
    packet_name = Column(Text, nullable=False)
    venv_id = Column(Integer, ForeignKey('venv.id'), nullable=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def build_as_code_first():
    """
    Function to create tables in the database and add default data to them.

    This function is called when the application starts and the database is empty.
    It creates all tables in the database and adds default data to them.
    """
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        db.add_all([
            JobStatus(id=0, status_name="builded", completed_status=0),
            JobStatus(id=1, status_name="running", completed_status=1),
            JobStatus(id=2, status_name="successed", completed_status=2),
            JobStatus(id=3, status_name="failed", completed_status=2),
            JobStatus(id=4, status_name="stoped", completed_status=2),
            Job(job_name="full_train", parent_job_id=None, status_id=None, venv_name="ai_venv"),
            Job(job_name="peft_train", parent_job_id=None, status_id=None, venv_name="ai_venv"),
        ])
        db.commit()


if __name__ == '__main__':
    build_as_code_first()