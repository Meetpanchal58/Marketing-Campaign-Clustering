echo "[$(date)]: START"

echo "[$(date)]: creating env with python 3.8 version"
python -m venv ./myenv

echo "[$(date)]: activating the environment"
source .\myenv\Scripts\activate

echo "[$(date)]: installing the dev requirements"
pip install -r requirements_dev.txt

echo "[$(date)]: END"