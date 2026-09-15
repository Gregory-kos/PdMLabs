class NotFitForSourceException(Exception):
    def __init__(self, message="'predict' was called on a model that was not previously fit for this source"):
        self.message = message
        super().__init__(self.message)


class IncompatibleMethodException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ShortScenarioLengthException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class CategoricalSpaceNotSupportedException(Exception):
    def __init__(self, message="The selected optimizer does not support categorical "
                               "(string/non-numeric) hyperparameter values. "
                               "Use a compatible optimizer or remove categorical parameters "
                               "from the hyperparameter space."):
        self.message = message
        super().__init__(self.message)