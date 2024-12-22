import inspect

__all__ = ["Functions"]


_TYPE_MAP ={
    "str": "string",
    "int": "integer"
}

class Functions:


    def __init__(self) -> None:
        self.functions = []
        self.maps = {}

    def tool(self, description: str, **kwargs):
        def decorator(func):
            # Extract the function name
            function_name = func.__name__

            # Extract function parameters
            parameters = {}
            sig = inspect.signature(func)
            required = []
            for param in sig.parameters.values():
                # Param description can be enhanced based on the type hint
                param_type = str(param.annotation.__name__) if param.annotation != param.empty else "string"
                parameters[param.name] = {
                    "type": _TYPE_MAP.get(param_type, param_type),
                    "description": kwargs.get(param.name, f"Parameter for {param.name}"),
                }
                if param.default != param.empty:
                    required.append(param.name)

            # Create OpenAI function schema
            function_schema = {
                "name": function_name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": list(parameters.keys())
                }
            }

            # Attach the generated schema to the function for later use
            self.functions.append(function_schema)
            self.maps[function_name] = func

            # Wrapper to call the function as normal
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper
        return decorator

    def get_functions(self, name:str):
        return self.maps[name]
