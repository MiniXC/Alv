from pathlib import Path


def class_with_path(Cls):
    old_init = Cls.__init__

    def new_init(self, *args, **kwargs):
        if "data_path" not in kwargs:
            raise ValueError("must pass data_path as keyword argument to class")
        self.data_path = kwargs["data_path"]
        Path(self.data_path).mkdir(parents=True, exist_ok=True)
        del kwargs["data_path"]
        old_init(self, *args, **kwargs)

    Cls.__init__ = new_init

    return Cls
