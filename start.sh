#!/bin/bash
export PYTHONUNBUFFERED=1
export TENSORFLOW_CPP_MIN_LOG_LEVEL=2
gunicorn --bind 0.0.0.0:10000 --workers 1 --threads 2 app:app 
