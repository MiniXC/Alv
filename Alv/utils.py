import os
from pathlib import Path
from glob import glob
import uuid

def class_with_path(prefix="alv_", delete_file_extension=""):
    def inner(Cls):
        old_init = Cls.__init__

        def new_init(self, *args, **kwargs):
            if "data_path" not in kwargs:
                raise ValueError("must pass data_path as keyword argument to class")
            self.data_path = kwargs["data_path"]
            Path(self.data_path).mkdir(parents=True, exist_ok=True)
            if delete_file_extension is not None:
                files = glob(os.path.join(self.data_path, f"{prefix}*.{delete_file_extension}"))
                for f in files:
                    os.remove(f)
            del kwargs["data_path"]
            old_init(self, *args, **kwargs)

        def generate_path(self):
            return os.path.join(self.data_path, f"{prefix}{uuid.uuid4().hex}.wav")

        Cls.__init__ = new_init
        Cls.generate_path = generate_path

        return Cls

    return inner
