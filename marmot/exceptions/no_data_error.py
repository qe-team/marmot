class NoDataError(Exception):

    def __init__(self, field, obj, module):
        message = "Missing field '" + field + "' in the object " + str(obj) + " needed in " + module
        super(NoDataError, self).__init__(message)
