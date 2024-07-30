from flytekit import workflow
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask

@workflow
def s3_list_objects_workflow() -> list:
    
    # Task to list objects in S3 bucket
    list_objects_task = DominoJobTask(
        name='List S3 objects',
        domino_job_config=DominoJobConfig(Command="python wineflow/list_objects.py"),
        inputs={},
        outputs={'objects': list},
        use_latest=True
    )
    
    objects = list_objects_task()

    return objects