from datetime import datetime
from pathlib import Path
from typing import List
import psutil
import subprocess
import time
from sqlalchemy.orm import Session
import threading
from entities import get_db, Process
from fine_tune_ops.crud import process_exit
from schemas import ProcessRegistryModel
import settings

class ProcessScanner:
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Singleton metaclass method for ensuring only one instance of ProcessScanner is created.

        :param args: Variable length argument list to pass to the constructor.
        :param kwargs: Keyword arguments to pass to the constructor.
        :return: The only instance of ProcessScanner.
        """
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance


    def __init__(self) -> None:
        threading.Thread.__init__(self)
        self.processes: List[ProcessRegistryModel] = []
    
    def run_process(self, db: Session, command:str, output_dir_path:str):
        """
        Runs a process in the given database session with the given command and output directory path.

        Args:
            db (Session): The database session to use.
            command (str): The command to run.
            output_dir_path (str): The directory path to write output files.

        Returns:
            subprocess.Popen: The process object.
        """
        now = datetime.now().isoformat().replace(':', '.')
        process = subprocess.Popen(command, shell=True, stdout=Path(output_dir_path) / f'process_output_{now}.log', stderr=f'process_error_{now}.log')
        self.add_process(process.pid)

    def add_process(self, *pids: ProcessRegistryModel):
        """
        Adds processes to the process list and starts a ScanWorker thread for each process.

        Args:
            *pids (ProcessRegistryModel): Variable length argument list of process registry models to add.
        """

        self.processes.extend(pids)
        print("PIDS eklendi:", pids)
        for pid in pids:
            ScanWorker(pid).start()
    
    def remove_process(self, pid:int):
        self.processes.remove(pid)



class ScanWorker(threading.Thread):

    def __init__(self, process:ProcessRegistryModel):
        threading.Thread.__init__(self)
        self.process = process
    
    def run(self):
        """
        Runs a ScanWorker thread for the given process.

        If the process is running, it waits for the process to finish and updates its exit code and end date in the database.
        If the process is not running (i.e., it has already finished), it sets the exit code to -1 and updates the end date in the database.
        """
        try:
            p = psutil.Process(self.process.pid)
            print(f"ScanWorker {self.process.pid} başlatıldı.")
            exit_code = p.wait()
            print(f"PID {self.process.pid} kapandı.")
            for db in get_db():
                process_exit(db, self.process, exit_code=exit_code, end_date=datetime.now())
        except psutil.NoSuchProcess:
            for db in get_db():
                process_exit(db, self.process, exit_code=-1, end_date=datetime.now())
