import os
from pathlib import Path

list_of_files = [
    '.github/workflows/.gitkeep',
    'src/__init__.py',
    'src/components/__init__.py',
    'src/components/data_ingestion.py',
    'src/components/data_tranformation.py',
    'src/components/model_trainer.py',
    'src/components/model_evaluation.py',
    'src/pipline/__init__.py',
    'src/pipline/training_pipline.py',
    'src/pipline/prediction_pipline.py',
    'src/utils/__init__.py',  # Added comma at the end
    'src/utils/utils.py',  # Fixed comma at the end
    'src/logger/__init__.py',  # Added directory __init__.py
    'src/logger/logging.py',  # Fixed comma at the end
    'src/exception/__init__.py',  # Added directory __init__.py
    'src/exception/exception.py',  # Fixed comma at the end
    'tests/unit/__init__.py',
    'tests/integration/__init__.py',
    'init_setup.sh',
    'requirements.txt',
    'requirements_dev.txt',
    'setup.py',
    'setup.config',
    'pyproject.toml',
    'tox.ini',
    'experiment/experiments.ipynb'
]

for file_path in list_of_files:
    if os.path.splitext(file_path)[1] == '':
        # This is a directory
        dir_path = Path(file_path)
        os.makedirs(dir_path, exist_ok=True)
    else:
        # This is a file
        file_path = Path(file_path)
        file_dir = file_path.parent
        os.makedirs(file_dir, exist_ok=True)
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            with open(file_path, "w") as f:
                pass  # create an empty file
