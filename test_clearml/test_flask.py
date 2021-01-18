import requests
from PIL import Image
from io import BytesIO
from flask import Flask, request
from clearml import Task
import matplotlib.pyplot as plt


app = Flask(__name__)

task = Task.init(project_name='pbl_example', task_name='flask', output_uri='http://180.76.145.171:30081/')
@app.route('/test', methods=['POST'])
def test():
    urls = request.json['urls']
    index = 121
    for url in urls:
        resp = requests.get(url, verify=False)
        index += 1
        # pillow读取
        pil = Image.open(BytesIO(resp.content))
        plt.imshow(pil)
        plt.axis('off')
        # cv2读取
        # src = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_UNCHANGED)
        # plt.imshow(src[:, :, ::-1])
        plt.savefig(f'./test{index}.png')
    return {'status': 'success'}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=32046)