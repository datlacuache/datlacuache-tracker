import prefect
from prefect import task
from prefect import Flow
import requests


@task
def test_api(url):
    """Task to test the API.

    Parameters
    ----------
    url : str
        The URL to test.
    """

    data = None

    logger = prefect.context.get('logger')
    logger.info('Testing API')

    response = requests.get(url)
    logger.info(f'Response: {response.status_code}')

    if response.status_code == 200:
        data = response.json()

    return data


@task
def return_activity(data):
    """Task to return the activity.

    Parameters
    ----------
    data : dict
        The data to return.
    """

    return data['activity']


with Flow('hello-flow') as flow:
    data = test_api('https://www.boredapi.com/api/activity')
    name = return_activity(data)

flow.register(project_name='test-prefect')
state = flow.run()

print(state.result.get(data)._result.value)
print(state.result.get(name)._result.value)
