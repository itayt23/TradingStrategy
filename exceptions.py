class ValueInsertionError(Exception):
    def __init__(self, message="Error in value insertion"):
        self.message = message
        super().__init__(self.message)

class ParameterError(Exception):
    def __init__(self, parameter_name, function,message="Incorrect parameter"):
        self.parameter_name = parameter_name
        self.message = f"Error in parameter '{parameter_name}': {message}\nfunction: {function}"
        super().__init__(self.message)

class FunctionNotValid(Exception):
    def __init__(self, lambda_function,function,message="Be aware of your indicator and the domain"):
        
        self.message = f"Error in lambda function :{lambda_function} {message}\nfunction: {function}"
        super().__init__(self.message)