from flytekit import task, workflow
from domino.data_sources import DataSourceClient

# Define the task
@task
def list_datasource_objects() -> list:
    # Instantiate a client and fetch the datasource instance
    object_store = DataSourceClient().get_datasource("winequalityworkshop")
    
    # List objects available in the datasource
    objects = object_store.list_objects()
    
    return objects

# Define the workflow
@workflow
def my_datasource_workflow() -> list:
    objects = list_datasource_objects()
    return objects

# Example execution (would typically be done by Flyte platform)
if __name__ == "__main__":
    result = my_datasource_workflow()
    print(result)