from flask import Flask, request, jsonify
from flask_cors import CORS
from classifier import URLNetClassifier  
import time

app = Flask(__name__)

CORS(app, resources={
    r"/detect": {
        "origins": [
            "https://hzj-2003.github.io",
            "https://c083-211-148-202-154.ngrok-free.app",
            "http://localhost:*"
        ],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})


print("[系统] 正在加载URL检测模型...")
start_load = time.time()
classifier = URLNetClassifier()  
load_time = time.time() - start_load
print(f"[系统] 模型初始化完成，耗时 {load_time:.2f}秒")

@app.route('/detect', methods=['POST'])
def detect():
    """与前端兼容的检测接口"""
    try:
        # 输入验证
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing required parameter: url"
            }), 400

        url = data['url'].strip()
        if not url:
            return jsonify({
                "status": "error",
                "message": "URL cannot be empty"
            }), 400

        
        start_time = time.time()
        raw_prediction = classifier.predict(url)  
        process_time = time.time() - start_time

       
        result_mapping = {
            "恶意": "phishing",
            "正常": "safe"
        }
        
    
        return jsonify({
            "result": result_mapping.get(raw_prediction, "unknown"),
            "probability": 0.95 if raw_prediction == "恶意" else 0.05,  
            "processing_time": process_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Detection failed: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
