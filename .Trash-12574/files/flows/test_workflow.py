from flytekit import workflow
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask

@workflow
def my_simple_test() -> str:

    # TASK ONE
    task_one = DominoJobTask(
        name="A small Hello",
        domino_job_config=DominoJobConfig(
            Command="python flows/print_hello.py"
        ),
        inputs={},
        outputs={'output': str},
        use_latest=True
    )
    output = task_one()
    return output
