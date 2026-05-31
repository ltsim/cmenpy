import functools


class ImprovementDeclaration:
    def __init__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            self.target = args[0]
            self.kwargs = {}
            if not isinstance(self.target, type):
                functools.update_wrapper(self, self.target)
        else:
            self.target = None
            self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        if self.target is not None:
            return self._execute(self.target, *args, **kwargs)

        target = args[0]

        @functools.wraps(target)
        def wrapper(*w_args, **w_kwargs):
            return self._execute(target, *w_args, **w_kwargs)

        if isinstance(target, type):
            return self._execute_class(target)

        return wrapper

    def _execute(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def _execute_class(self, cls):
        return cls
