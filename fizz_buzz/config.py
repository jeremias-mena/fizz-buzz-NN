import os

class Config:
    def __init__(self):
        pass

    def create_dir(self, dir_path: str) -> None:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        else:
            print(f"El directorio {dir_path} ya existe")
        return