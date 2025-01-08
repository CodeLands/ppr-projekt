# app/integration/tests/test_faces.py

def test_register_face(client):
    response = client.post('/register-face', data=b'some binary data', content_type='application/octet-stream')
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}, Response body: {response.data}"

def test_verify_face(client):
    response = client.post('/login-face', data=b'some binary data', content_type='application/octet-stream')  # Correct route used here
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}, Response body: {response.data}"
