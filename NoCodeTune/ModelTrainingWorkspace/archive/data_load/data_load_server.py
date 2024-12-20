import sys
from fastapi import FastAPI, HTTPException, UploadFile ,File,Request,status
import pandas as pd
import os
try :
    from file_operations_utils import *
except:
    from data_load.file_operations_utils import *    
import requests
from pathlib import Path
from typing import List, Union
from pydantic import BaseModel
from uuid import uuid4
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import shutil
from typing import Annotated
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from FineTuneFlow.logs import Logger
class Response(BaseModel):
    text: str = ""
    is_success: bool = True


class ListResponse(BaseModel):
    content: list = []
    is_success: bool = True
    error_code :int = 200

class Params(BaseModel):
    path :str= ""
    columns :list = None

app = FastAPI()
BASE_PATH = "DataLoad/Files/"
os.makedirs(BASE_PATH, exist_ok= True)


#post /files/upload yeni bir dosyanın eklenmesi
#post /files/{id/upload/ igili dosyaya yeni versiyon eklenmesi
@app.post("/file")
async def file_upload(data_files: Annotated[
        list[UploadFile], File(description="Multiple files as UploadFile")
    ], task_type:str, tags:list) -> Response:
    print("----",task_type)
    print("----",tags)
    file_names = []
    file_paths = []
    response_text = "Files uploaded successfully:\n"
    exist_files_error = "No files could be uploaded.\n Below files are conflict:\n"
    are_exists_files = [str(Path(BASE_PATH) / data_file.filename) for data_file in data_files if os.path.isfile(str(Path(BASE_PATH) / data_file.filename) )]

    if are_exists_files and len(are_exists_files)>0:
        exist_files_error += str(are_exists_files)
        Logger().info(message=response_text)
        return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content=jsonable_encoder(Response(text=exist_files_error ,is_success=False)),
    )
    for data_file in data_files:
        file_path = str(Path(BASE_PATH) / data_file.filename)
        os.makedirs(file_path,exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(data_file.file, buffer)
            file_names.append(data_file.filename)
            file_paths.append(file_path) 
            response_text += f"'{data_file.filename}' at '{file_path}'\n"
        Logger().info(message=response_text)
    return Response(text=response_text , is_success=True)

#get /files mevcuttaki tüm ana dosyaların isimlerini getirir
#get /files/{id} belirtilen id deki dosyanın içeriğini getirir
#put /files/{id} belirtilen id deki dosyanın meta datasını günceller
#get files/{id}/versions belirtilen iddeki tüm versiyonları getirir
#get files/{id}/versions/{Vid} belirtilen iddeki  versiyonları getirir
@app.get("/files")
async def file_read( params:Params)-> Response|ListResponse:

    """
    Retrieve the contents of a file or list all files if no path is specified.

    Args:
    - params: An instance of Params containing the file path and optional columns.

    Returns:
    - A Response with file contents if the path is valid.
    - A ListResponse with all file names if no path is provided.
    - A JSONResponse with an error message if the path is invalid or an error occurs.

    Raises:
    - HTTP_404_NOT_FOUND: If the file path is not specified or doesn't exist.
    - HTTP_405_METHOD_NOT_ALLOWED: If an exception occurs during file reading.
    """

    if not params.path:
        #TODO: Add 209 warning  
        return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=jsonable_encoder(ListResponse(content= os.listdir(BASE_PATH), is_success=True,error_code = 209)),
    )
        
    elif not os.path.isfile(params.path):
        Logger().error(message=f"{params.path} is not found")
        return  JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=jsonable_encoder(Response(text=f"{params.path} is not found" ,is_success=False)),
    )
    try:
        text = read_textual_file(params.path,params.columns)
        return  Response(text=text ,is_success=True)
    except Exception as e:
        return  JSONResponse(
        status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
        content=jsonable_encoder(Response(text=f"{e}" , is_success=False)),
    )
    

#delete /files/{id} belirtilen id deki dosyayı siler
#delete files/{id}/versions/{Vid} belirtilen iddeki  versiyonu siler
@app.delete("/files/{path}")
async def file_delete(path:str)-> Response:
    # modellleri silerken alt kolasöreleri sil ana dizini silme örn:  ~V1_2 olarak güncelle alt dosyaları sil
    if not os.path.isfile(path):
        Logger().error(message=f"{path} is not found")
        #raise "Error"
        return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=jsonable_encoder(Response(text=f"{path} is not found" ,is_success=False)),
    )
    try:
        os.remove(path)
        Logger().info(message=f"{path} has been deleted successfully")
        return Response( text=f"{path} has been deleted successfully", is_success=True)
    except OSError as e:
        Logger().error(message=f"{path} - {e.strerror}")
        return  JSONResponse(
        status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
        content=jsonable_encoder(Response(text=f"{e}" , is_success=False)),
    )

    




#response = requests.post("http://192.168.20.106:8010/file/read", params=params)
#print(response.json())