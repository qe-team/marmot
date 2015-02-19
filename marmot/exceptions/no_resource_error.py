class NoResourceError(Exception):
    def __init__(self, resource, module):
        message = "No " + resource + " provided in " + str(module)
        super(NoResourceError, self).__init__(message)
