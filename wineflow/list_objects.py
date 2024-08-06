from contextlib import redirect_stdout
from domino.data_sources import DataSourceClient

# Define the file path where you want to redirect the output
file_path = "/workflow/outputs/objects"

# Open the file in write mode and use redirect_stdout to redirect print output
with open(file_path, 'w') as f:
    with redirect_stdout(f):
        # Instantiate a client and fetch the datasource instance
        object_store = DataSourceClient().get_datasource("winequality")
        
        # List objects available in the datasource
        objects = object_store.list_objects()
        
        # Print objects to be redirected to the file
        print(objects)