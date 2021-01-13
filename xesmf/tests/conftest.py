import dask
import pytest


@pytest.fixture(scope='function')
def threaded_scheduler():
    with dask.config.set(scheduler='threads'):
        yield


@pytest.fixture(scope='function')
def processes_scheduler():
    with dask.config.set(scheduler='processes'):
        yield


@pytest.fixture(scope='module')
def distributed_scheduler():
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(threads_per_worker=1, n_workers=2, processes=True)
    client = Client(cluster)
    yield
    client.close()
    del client
    cluster.close()
    del cluster


def pytest_addoption(parser):
    parser.addoption('--runtestcases', action='store_true', default=False, help='run test cases')


def pytest_configure(config):
    config.addinivalue_line('markers', 'testcases: mark test cases')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--runtestcases'):
        # --runtestcases given in cli: do not skip test cases
        return
    skip_testcases = pytest.mark.skip(reason='need --runtestcases option to run')
    for item in items:
        if 'testcases' in item.keywords:
            item.add_marker(skip_testcases)
