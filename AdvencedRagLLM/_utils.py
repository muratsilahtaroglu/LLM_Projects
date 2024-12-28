import time
import pandas as pd
class CalculateTime:
    def __init__(self, info:str,file) -> None:
        self.info = info
        self.file = file
    
    def __enter__(self):
        self.st = time.time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.et = time.time()
        elapsed_time = self.et - self.st
        print(f'\n{self.info} runtime:', elapsed_time, 'seconds',file=self.file)
        print(f'\n{self.info} runtime:', elapsed_time/60, 'minute',file=self.file)
        print(f'\n{self.info} runtime:', elapsed_time/(60*60), 'hour\n',file=self.file)

