import logging
import os
from datetime import datetime


# Creating logs directory to store log in files
LOG_DIR = "logs"
LOG_DIR = os.path.join(os.getcwd(), LOG_DIR)

# Creating LOG_DIR if it does not exists.
os.makedirs(LOG_DIR, exist_ok=True)


# Creating file name for log file based on current timestamp
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
file_name = f"log_{CURRENT_TIME_STAMP}.log"

# Creating file path for projects.
log_file_path = os.path.join(LOG_DIR, file_name)


# Configure logging
logging.basicConfig(
    filename=log_file_path,
    filemode='w',
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Also add console handler for immediate feedback
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Get the root logger and add console handler
logger = logging.getLogger()
logger.addHandler(console_handler)

# Export the logging object
logging = logger