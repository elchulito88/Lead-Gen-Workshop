from domino.data_sources import DataSourceClient

# Instantiate a client and fetch the datasource instance
object_store = DataSourceClient().get_datasource("winequality")

# List objects available in the datasource
objects = object_store.list_objects()

# Ensure the output is printed or logged in a way that Domino can capture it
print(objects)