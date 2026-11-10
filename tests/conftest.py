"""Shared fixtures for the ADR test suite."""

import pytest
from concurrent.futures import ProcessPoolExecutor

from radical.asyncflow import WorkflowEngine, LocalExecutionBackend


@pytest.fixture
async def engine():
    backend = await LocalExecutionBackend(ProcessPoolExecutor(max_workers=2))
    eng = await WorkflowEngine.create(backend)
    yield eng
    await eng.shutdown()
