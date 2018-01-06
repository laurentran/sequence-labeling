# Spark configuration and packages specification. The dependencies defined in
# this file will be automatically provisioned for each run that uses Spark.

import sys
import os
import argparse

from azureml.logging import get_azureml_logger

# initialize the logger
logger = get_azureml_logger()

# add experiment arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--arg', action='store_true', help='My Arg')
args = parser.parse_args()
print(args)

# This is how you log scalar metrics
# logger.log("MyMetric", value)

# Create the outputs folder - save any outputs you want managed by AzureML here
os.makedirs('./outputs', exist_ok=True)