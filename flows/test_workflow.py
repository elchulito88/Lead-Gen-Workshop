from flytekit import workflow
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask

@workflow
def my_simple_test() -> str:

    # TASK ONE
    task_one = DominoJobTask(
        name="List files in directory",
        domino_job_config=DominoJobConfig(Command="ls -ltrash > /workflow/outputs/output"),
        inputs={},
        outputs={'output': str},
        use_latest=True
    )
    output = task_one()
    return output