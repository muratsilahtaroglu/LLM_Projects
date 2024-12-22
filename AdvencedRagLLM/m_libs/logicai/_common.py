__all__ = ['AlgoException']

class AlgoException(BaseException):

    def __init__(self, *args):
        super().__init__(*args)