import os
import datetime
import json
import inspect
from colorama import Fore, Style

__all__ = ["ChatLogger"]

def calling_function_full_name(stack_level: int = 1):
    """
    Returns the full name of the calling function, including module and class names.

    This function inspects the call stack to determine the full name of the
    function at a specified stack level. It includes the module name, class
    name (if available), and function name.

    Args:
        stack_level (int): The level in the call stack to inspect. Defaults to 1.

    Returns:
        str: A string representing the full name of the calling function in
             the format "<module>.<class>.<function>". If the class name is not
             available, it returns "<module>.<function>".
    """
    frame = inspect.stack()[stack_level]
    module = inspect.getmodule(frame[0])
    
    if module is not None:
        module_name = module.__name__
    else:
        module_name = "<module>"
    
    func_name = frame.function
    # 'self' ya da 'cls' var mı kontrol ediyoruz
    class_name = None
    if 'self' in frame.frame.f_locals:
        class_name = frame.frame.f_locals['self'].__class__.__name__
    elif 'cls' in frame.frame.f_locals:
        class_name = frame.frame.f_locals['cls'].__name__
    
    # class_name varsa onu da dahil ediyoruz
    if class_name:
        return f"{module_name}.{class_name}.{func_name}"
    else:
        return f"{module_name}.{func_name}"


class ChatLogger:

    def __init__(self, log_folder:str|None) -> None:
        self.log_folder = log_folder
        self._log_day = None
        self._file = None
        self.log_file_manager()
    
    def log_file_manager(self):
        if self.log_folder:
            if self._log_day != datetime.date.today():
                self._log_day = datetime.date.today()
                if self._file: self._file.close()
                self._log_file_name = f"chat-{self._log_day.isoformat()}.log"
                self._file = open(os.path.join(self.log_folder, self._log_file_name), 'a', encoding="utf-8")
        
         
    def _write(self, **kwargs):
        if self.log_folder:
            self.log_file_manager()
            date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            kwargs.update({"date": date, "pid": os.getpid()})
            self._file.write(json.dumps(kwargs, ensure_ascii=False) + "\n")
            self._file.flush()
    
    def debug(self, topic:str, **kwargs):
        kwargs.update({"topic": topic,"level": "debug", "source_function": calling_function_full_name(2)})
        self._write(**kwargs)

    def info(self, topic:str, **kwargs):
        kwargs.update({"topic": topic, "level": "info", "source_function": calling_function_full_name(2)})
        self._write(**kwargs)

    def warning(self, topic:str, **kwargs):
        kwargs.update({"topic": topic,"level": "warning", "source_function": calling_function_full_name(2)})
        self._write(**kwargs)

    def error(self, topic:str, **kwargs):
        print(Fore.RED + "ERROR: " + topic + "\n" + str(kwargs) + Style.RESET_ALL)
        kwargs.update({"topic": topic,"level": "error", "source_function": calling_function_full_name(2)})
        self._write(**kwargs)
    
    def critical(self, topic:str, **kwargs):
        print(Fore.RED + "ERROR: " + topic + "\n" + str(kwargs) + Style.RESET_ALL)
        kwargs.update({"topic": topic,"level": "critical", "source_function": calling_function_full_name(2)})
        self._write(**kwargs)
    
    def close(self) -> None:
        if self.log_folder and self._file.closed is False:
            self._file.close()
