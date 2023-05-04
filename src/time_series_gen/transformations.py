class Transformation:
    def __init__(self, **kwargs):
        self.params = kwargs

    def apply(self, series):
        raise NotImplementedError()

class ConstantOffsetTransformation(Transformation):
    def apply(self, series):
        offset = self.params.get('offset', 0)
        return series + offset

class NonLinearTransformation(Transformation):
    def apply(self, series):
        function = self.params.get('function', lambda x: x)
        return function(series)
