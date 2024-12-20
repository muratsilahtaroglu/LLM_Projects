from fastapi import HTTPException, status, Request
from fastapi.responses import JSONResponse

class ConflictError(Exception):
    """Custom Conflict Error"""
    def __init__(self, message="A conflict occurred"):
        self.message = message
        super().__init__(self.message)

class NoContentError(Exception):
    """Custom No Content Error"""
    def __init__(self, message="No content available"):
        self.message = message
        super().__init__(self.message)

class FileSystemError(Exception):
    """File System Error"""
    def __init__(self, message="File system error"):
        self.message = message
        super().__init__(self.message)
    
class NotValidError(Exception):
    """Custom Not Content Error"""
    def __init__(self, message="Not valid!"):
        self.message = message
        super().__init__(self.message)

class AIServerError(Exception):
    """General Error"""
    def __init__(self, message="Error!"):
        self.message = message
        super().__init__(self.message)


def config_exception_handler(app):
    @app.exception_handler(ConflictError)
    async def ConflictError_handler(request: Request, exc: ConflictError):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"message": exc.message},
        )
    @app.exception_handler(NoContentError)
    async def NotContentError_handler(request: Request, exc: NoContentError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": exc.message},
        )
    @app.exception_handler(NotValidError)
    async def NotValidError_handler(request: Request, exc: NotValidError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"message": exc.message},
        )
    @app.exception_handler(FileSystemError)
    async def FileSystemError_handler(request: Request, exc: FileSystemError):
        return JSONResponse(
            status_code=512,
            content={"message": exc.message},
        )
    @app.exception_handler(NotImplementedError)
    async def NotImplementedError_handler(request: Request, exc: NotImplementedError):
        return JSONResponse(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            content={"message": exc.args},
        )
    @app.exception_handler(AIServerError)
    async def AIServerError_handler(request: Request, exc: AIServerError):
        return JSONResponse(
            status_code=513,
            content={"message": exc.args},
        )