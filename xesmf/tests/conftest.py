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
