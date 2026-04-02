import logging
import os
from datetime import datetime

# Create log filename with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create logs/ directory if it doesn't exist
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=logs_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
