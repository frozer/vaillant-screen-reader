import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from markup_helper.api.main import create_app
from markup_helper.api.store import ReviewStore

FIXTURE = Path(__file__).parent / 'fixtures' / 'sample.csv'


@pytest.fixture
def client(tmp_path):
    labels = tmp_path / 'labels.csv'
    shutil.copy(FIXTURE, labels)
    images = tmp_path / 'images'
    images.mkdir()
    (images / 'cam01-snapshot-20261022074502.jpg').write_bytes(b'\xff\xd8fake')
    static = tmp_path / 'dist'
    static.mkdir()
    (static / 'index.html').write_text('<html>app</html>')
    store = ReviewStore(labels, tmp_path / 'out.csv', images)
    return TestClient(create_app(store, static_dir=static))


def test_rows_and_meta(client):
    rows = client.get('/api/rows').json()
    assert len(rows) == 8
    assert rows[1]['recognized']['temperature'] == 37
    assert rows[1]['image_exists'] is True
    assert rows[0]['image_exists'] is False
    assert client.get('/api/meta').json()['counts']['pending'] == 8


def test_image(client):
    assert client.get('/api/rows/1/image').content == b'\xff\xd8fake'
    assert client.get('/api/rows/0/image').status_code == 404
    assert client.get('/api/rows/99/image').status_code == 404


def test_review_flow(client):
    r = client.post('/api/rows/1/review', json={'action': 'too_dark'})
    assert r.status_code == 200
    assert r.json()['retrain_needed'] == 'ScreenState:ok>unreadable'

    r = client.post('/api/rows/1/review', json={
        'action': 'update',
        'values': {'temperature': 36, 'burn': False, 'heating': True, 'hotwater': False,
                   'pump': False, 'gasvalve': False}})
    assert r.json()['status'] == 'updated'
    assert r.json()['retrain_needed'] == 'Temperature:37>36'

    assert client.post('/api/rows/1/reset').json()['status'] == 'pending'


def test_review_validation(client):
    assert client.post('/api/rows/1/review', json={'action': 'nope'}).status_code == 422
    assert client.post('/api/rows/1/review', json={'action': 'update'}).status_code == 422
    r = client.post('/api/rows/1/review', json={
        'action': 'update', 'values': {'temperature': 500}})
    assert r.status_code == 422
    assert client.post('/api/rows/42/review', json={'action': 'confirm'}).status_code == 404


def test_static_mount(client):
    assert 'app' in client.get('/').text
