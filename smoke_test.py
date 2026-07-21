import io
import sys
import requests
from PIL import Image

# Create a simple gray test image
img = Image.new('RGB', (224, 224), (128, 128, 128))
buf = io.BytesIO()
img.save(buf, 'JPEG')
buf.seek(0)

files = {'file': ('test.jpg', buf, 'image/jpeg')}

try:
    r = requests.post('http://127.0.0.1:5000/predict', files=files, timeout=15)
    print('STATUS', r.status_code)
    print('URL', r.url)
    print('BODY_START')
    print(r.text[:2000])
    print('\nBODY_END')
except Exception as e:
    print('ERROR', e)
    sys.exit(2)
