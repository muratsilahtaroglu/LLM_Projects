import json
import os
import numpy as np
import pandas as pd
import charset_normalizer
import docx
import markdown
import PyPDF2
import yaml
from bs4 import BeautifulSoup

from langchain_community.document_loaders import UnstructuredPDFLoader
from pylatexenc.latex2text import LatexNodes2Text
from pydantic import BaseModel

from logs import Logger



class DataFrameRequest:
    data: pd.DataFrame

class ParserStrategy:
    def read(self, file_path: str) -> str:
        raise NotImplementedError


#Excel 
class ExcelParser(ParserStrategy):
    def read(self, file_path: str) -> str:
        df = pd.read_excel(file_path, index_col=False)
        return df

class ParquetParser(ParserStrategy):
    def read(self, file_path: str) -> str:
        df = pd.read_parquet(file_path)
        return df
# Basic text file reading
class TXTParser(ParserStrategy):
    def read(self, file_path: str) -> str:
        charset_match = charset_normalizer.from_path(file_path).best()
        #logger.debug(f"Reading '{file_path}' with encoding '{charset_match.encoding}'")
        return str(charset_match)


# Reading text from binary file using pdf parser
class PDFParser(ParserStrategy):
    # def read(self, file_path: str) -> str:
    #     parser = PyPDF2.PdfReader(file_path)
    #     text = ""
    #     for page_idx in range(len(parser.pages)):
    #         text += parser.pages[page_idx].extract_text()
    #     return text
    
    def read(self,file_path: str) -> str:
        """
        Reads the content of a pdf file and returns it as a string.

        Args:
            file_path (str): The path to the file to be read.

        Returns:
            str: The content of the file.
        """
        loader = UnstructuredPDFLoader(file_path,mode="elements",strategy="hi_res",infer_table_structure=True,extract_images_in_pdf=True)
        data = loader.load()
        text = ""
        for page_idx in range(len(data)):
            text += data[page_idx].page_content + '\n'
        return text
    

# Reading text from binary file using docs parser
class DOCXParser(ParserStrategy):
    def read(self, file_path: str) -> str:
        doc_file = docx.Document(file_path)
        text = ""
        for para in doc_file.paragraphs:
            text += para.text
        return text


# Reading as dictionary and returning string format
class JSONParser(ParserStrategy):
    def read(self, file_path: str) -> str:
        with open(file_path, "r") as f:
            data = json.load(f)
            text = str(data)
        return text


class XMLParser(ParserStrategy):
    def read(self, file_path: str) -> str:
        with open(file_path, "r") as f:
            soup = BeautifulSoup(f, "xml")
            text = soup.get_text()
        return text


# Reading as dictionary and returning string format
class YAMLParser(ParserStrategy):
    def read(self, file_path: str) -> str:
        with open(file_path, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            text = str(data)
        return text


class HTMLParser(ParserStrategy):
    def read(self, file_path: str) -> str:
        with open(file_path, "r") as f:
            soup = BeautifulSoup(f, "html.parser")
            text = soup.get_text()
        return text

class MarkdownParser(ParserStrategy):
    def read(self, file_path: str) -> str:
        with open(file_path, "r") as f:
            html = markdown.markdown(f.read())
            text = "".join(BeautifulSoup(html, "html.parser").findAll(string=True))
        return text


class LaTeXParser(ParserStrategy):
    def read(self, file_path: str) -> str:
        with open(file_path, "r") as f:
            latex = f.read()
        text = LatexNodes2Text().latex_to_text(latex)
        return text


class FileContext:
    def __init__(self, parser: ParserStrategy):
        self.parser = parser
        #self.logger = logger

    def set_parser(self, parser: ParserStrategy) -> None:
        #self.logger.debug(f"Setting Context Parser to {parser}")
        self.parser = parser

    def read_file(self, file_path) -> str:
        #self.logger.debug(f"Reading file {file_path} with parser {self.parser}")
        return self.parser.read(file_path)

#TODO: add more parsers
extension_to_parser = {
    ".txt": TXTParser(),
    ".csv": TXTParser(),
    ".pdf": PDFParser(),
    ".docx": DOCXParser(),
    ".json": JSONParser(),
    ".xml": XMLParser(),
    ".yaml": YAMLParser(),
    ".yml": YAMLParser(),
    ".html": HTMLParser(),
    ".htm": HTMLParser(),
    ".xhtml": HTMLParser(),
    ".md": MarkdownParser(),
    ".markdown": MarkdownParser(),
    ".tex": LaTeXParser(),
    ".xlsx": ExcelParser(),
    ".parquet": ParquetParser()
}


def is_file_binary_fn(file_path: str):
    """Given a file path load all its content and checks if the null bytes is present

    Args:
        file_path (_type_): _description_

    Returns:
        bool: is_binary
    """
    with open(file_path, "rb") as f:
        file_data = f.read()
    if b"\x00" in file_data:
        return True
    return False


def read_textual_file(file_path: str) -> str:
    """
    Read textual file content based on file extension.
    Supported extensions are:
        - .txt
        - .csv
        - .pdf
        - .docx
        - .json
        - .xml
        - .yaml
        - .yml
        - .html
        - .htm
        - .xhtml
        - .md
        - .markdown
        - .tex
        - .xlsx
        - .parquet
    If the file is binary, it will raise ValueError with "Unsupported binary file format".
    If the file is not found, it will raise FileNotFoundError with "read_file {file_path} failed: no such file or directory".
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"read_file {file_path} failed: no such file or directory"
        )
    is_binary = is_file_binary_fn(file_path)
    file_extension = os.path.splitext(file_path)[1].lower()
    parser = extension_to_parser.get(file_extension)
    if not parser:
        if is_binary:
            raise ValueError(f"Unsupported binary file format: {file_extension}")
        # fallback to txt file parser (to support script and code files loading)
        parser = TXTParser()
    file_context = FileContext(parser)
    return file_context.read_file(file_path)

def convert_to_json_serializable(obj):
    """
    Convert an object to a type that can be serialized to JSON.

    This function converts certain types of objects to types that can be serialized to JSON.
    The types that are converted are:

    - NumPy integers: converted to Python int
    - NumPy floats: converted to Python float
    - NumPy arrays: converted to lists

    If the object is not one of the above types, a TypeError is raised with a message
    indicating the type of the object.

    Parameters
    ----------
    obj : object
        The object to be converted.

    Returns
    -------
    object
        The converted object.

    Raises
    ------
    TypeError
        If the object is not one of the above types.
    """
    if isinstance(obj, np.integer):  
        return int(obj)
    elif isinstance(obj, np.floating):  
        return float(obj)
    elif isinstance(obj, np.ndarray):  
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

def write_textual_file(data: str | list, path: str):

    """
    Writes a textual file given a path and data.
    
    Parameters
    ----------
    data : str or list
        The data to be written to the file. If a list, each element will be written as a line.
    path : str
        The file path to write to.
    
    Raises
    ------
    ValueError
        If the file path does not end with a supported extension (.json, .csv, .xlsx, .txt).
    """
    import pandas as pd  

    if path.endswith('.json'):
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4,default=convert_to_json_serializable)
        print("File saved:", path)
    elif path.endswith('.csv'):
        df = pd.DataFrame(data)
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print("File saved:", path)
    elif path.endswith('.xlsx'):
        df = pd.DataFrame(data)
        df.to_excel(path, index=False)
        print("File saved:", path)
    elif path.endswith('.txt'):
        with open(path, "w", encoding='utf-8') as file:
            if isinstance(data, list):
                for line in data:
                    file.write(str(line) + '\n')
            else:
                file.write(data)
        print("File saved:", path)
    else:
        print("Unsupported file extension.")
