from flytekit import workflow
from flytekitplugins.domino.task import DominoJobConfig, DominoJobTask

@workflow
def etl_workflow(a: int, b: int) -> (int, int, int, str, str, int):
    # Create addition task
    add_task = DominoJobTask(
        name='Add Data',
        domino_job_config=DominoJobConfig(Command="python add.py"),
        inputs={'first_value': int, 'second_value': int},
        outputs={'sum': int},
        use_latest=True
    )
    sum_result = add_task(first_value=a, second_value=b)

    # Create multiplication task
    multiply_task = DominoJobTask(
        name='Check numbers',
        domino_job_config=DominoJobConfig(Command="python multiply.py"),
        inputs={'first_value': int, 'second_value': int},
        outputs={'product': int},
        use_latest=True
    )
    product_result = multiply_task(first_value=a, second_value=b)

    # Create subtraction task that depends on the addition task
    subtract_task = DominoJobTask(
        name='Remove data',
        domino_job_config=DominoJobConfig(Command="python subtract.py"),
        inputs={'first_value': int, 'second_value': int},
        outputs={'difference': int},
        use_latest=True
    )
    difference_result = subtract_task(first_value=sum_result, second_value=b)

    # Create division task that runs in parallel with the subtraction task
    divide_task = DominoJobTask(
        name='Sort the data',
        domino_job_config=DominoJobConfig(Command="python divide.py"),
        inputs={'first_value': int, 'second_value': int},
        outputs={'division': str},
        use_latest=True
    )
    division_result = divide_task(first_value=a, second_value=b)

    # Create modulus task that runs in parallel with the subtraction and division tasks
    modulus_task = DominoJobTask(
        name='Line it all up',
        domino_job_config=DominoJobConfig(Command="python modulus.py"),
        inputs={'first_value': int, 'second_value': int},
        outputs={'modulus': str},
        use_latest=True
    )
    modulus_result = modulus_task(first_value=a, second_value=b)

    # Create an aggregation task that depends on the sum, difference, and product tasks
    aggregate_task = DominoJobTask(
        name='Aggregate results',
        domino_job_config=DominoJobConfig(Command="python aggregate.py"),
        inputs={'sum': int, 'difference': int, 'product': int},
        outputs={'aggregate': int},
        use_latest=True
    )
    aggregate_result = aggregate_task(sum=sum_result, difference=difference_result, product=product_result)

    # Return the results as separate outputs
    return sum_result, product_result, difference_result, division_result, modulus_result, aggregate_result