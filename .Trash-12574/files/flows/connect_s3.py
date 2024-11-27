from domino.data_sources import DataSourceClient

# instantiate a client and fetch the datasource instance
object_store = DataSourceClient().get_datasource("winequality")

# list objects available in the datasource
objects = object_store.list_objects()