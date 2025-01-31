import logging
import os
from datetime import datetime

def setup_logging(log_folder='log'):
    # Create the log folder if it doesn't exist
    os.makedirs(log_folder, exist_ok=True)
    
    # Generate a timestamp for the log file name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'model_log.log'
    log_path = os.path.join(log_folder, log_file)
    
    # Set up logging configuration
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )