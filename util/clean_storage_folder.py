import shutil
from . import make_storage_folder

def clean_storage(storage_directory):
    shutil.rmtree(storage_directory)
    make_storage_folder.make_storage_folder(storage_directory)

def clean_logs(log_directory):
    try:
        shutil.rmtree(log_directory)
    except FileNotFoundError:
        pass
    make_storage_folder.make_logs_folder(log_directory)