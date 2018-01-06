# This script generates the scoring and schema files
# necessary to operationalize your model
from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema
from azureml.assets import get_local_path

# Prepare the web service definition by authoring
# init() and run() functions. Test the functions
# before deploying the web service.

model = None

def init():
    # Get the path to the model asset
    # local_path = get_local_path('mymodel.model.link')
    
    # Load model using appropriate library and function
    global model
    # model = model_load_function(local_path)
    model = 42

def run(input_df):
    import json
    
    # Predict using appropriate functions
    # prediction = model.predict(input_df)

    prediction = "%s %d" % (str(input_df), model)
    return json.dumps(str(prediction))

def generate_api_schema():
    import os
    print("create schema")
    sample_input = "sample data text"
    inputs = {"input_df": SampleDefinition(DataTypes.STANDARD, sample_input)}
    os.makedirs('outputs', exist_ok=True)
    print(generate_schema(inputs=inputs, filepath="outputs/schema.json", run_func=run))

# Implement test code to run in IDE or Azure ML Workbench
if __name__ == '__main__':
    # Import the logger only for Workbench runs
    from azureml.logging import get_azureml_logger

    logger = get_azureml_logger()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true', help='Generate Schema')
    args = parser.parse_args()

    if args.generate:
        generate_api_schema()

    init()
    input = "{}"
    result = run(input)
    logger.log("Result",result)
