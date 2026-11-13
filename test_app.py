import io
from app import app

def run_tests():
    client = app.test_client()

    print('=== Test 1: Upload Brain Tumor Image (Form) ===')
    with open('upload/Brain Tumor/image.png', 'rb') as f:
        data = {'imagefile': (io.BytesIO(f.read()), 'tumor_test.png')}
        r = client.post('/upload', data=data, content_type='multipart/form-data')
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    html = r.get_data(as_text=True)
    assert 'Abnormality Detected' in html, 'Expected Abnormality Detected in HTML'
    assert 'Brain Tumor' in html, 'Expected Brain Tumor in HTML'
    assert 'data:image/png;base64,' in html, 'Expected base64 image in HTML'
    print('Test 1 Passed: Tumor image correctly classified in HTML!')

    print('=== Test 2: Upload Normal Image (Form) ===')
    with open('upload/Normal/image.png', 'rb') as f:
        data = {'imagefile': (io.BytesIO(f.read()), 'normal_test.png')}
        r = client.post('/upload', data=data, content_type='multipart/form-data')
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    html = r.get_data(as_text=True)
    assert 'No Tumor Detected' in html, 'Expected No Tumor Detected in HTML'
    assert 'Normal' in html, 'Expected Normal in HTML'
    print('Test 2 Passed: Normal image correctly classified in HTML!')

    print('=== Test 3: API Predict Tumor (JSON) ===')
    with open('upload/Brain Tumor/image.png', 'rb') as f:
        data = {'file': (io.BytesIO(f.read()), 'tumor_test.png')}
        r = client.post('/api/predict', data=data, content_type='multipart/form-data', headers={'Accept': 'application/json'})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    res = r.get_json()
    assert res['success'] == True
    assert res['is_tumor'] == True
    assert 'Brain Tumor' in res['prediction']
    print(f"Test 3 Passed! API JSON response: {res['prediction']} ({res['confidence']}%)")

    print('=== Test 4: API Predict Normal (JSON) ===')
    with open('upload/Normal/image.png', 'rb') as f:
        data = {'file': (io.BytesIO(f.read()), 'normal_test.png')}
        r = client.post('/api/predict', data=data, content_type='multipart/form-data', headers={'Accept': 'application/json'})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    res = r.get_json()
    assert res['success'] == True
    assert res['is_tumor'] == False
    assert 'Normal' in res['prediction']
    print(f"Test 4 Passed! API JSON response: {res['prediction']} ({res['confidence']}%)")

    print('=== Test 5: Empty file upload ===')
    r = client.post('/upload', data={}, content_type='multipart/form-data')
    assert 'Please choose a valid MRI image file!' in r.get_data(as_text=True)
    print('Test 5 Passed: Empty file handled gracefully!')

    print('ALL PREDICTION AND UPLOAD TESTS PASSED SUCCESSFULLY!')

if __name__ == '__main__':
    run_tests()

