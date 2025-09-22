"""
Logging configuration for ErgoSense.
Ensures all logs are properly suppressed.
"""

import logging
import os
import warnings

def configure_silent_logging():
    # Suppress all warnings
    warnings.filterwarnings('ignore')
    
    # Configure root logger
    logging.getLogger().setLevel(logging.CRITICAL)
    
    # Suppress specific loggers
    LOGGERS_TO_DISABLE = [
        'mediapipe',
        'mediapipe.python',
        'tensorflow',
        'absl',
        'matplotlib',
        'PIL',
        'h5py',
        'numba',
        'streamlit',
    ]
    
    for logger_name in LOGGERS_TO_DISABLE:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
    
    # Environment variables for C++ logs
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
    os.environ["METAL_DEVICE_WRAPPER_TYPE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Disable absl logging
    try:
        import absl.logging
        absl.logging.set_verbosity(absl.logging.FATAL)
        absl.logging.use_absl_handler()
    except ImportError:
        pass
