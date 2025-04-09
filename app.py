from flask import Flask, render_template, request, jsonify
from classifier import URLNetClassifier
import os
import time

app = Flask(__name__)

# 初始化分类器（约需2-5秒）
print("[系统] 正在加载URL检测模型...")
start_time = time.time()
classifier = URLNetClassifier()
load_time = time.time() - start_time
print(f"[系统] 模型加载完成，耗时 {load_time:.2f} 秒")

@app.route('/')
def home():
    """渲染检测页面"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """处理检测请求"""
    try:
        # 获取并验证输入
        url = request.json.get('url', '').strip()
        if not url:
            return jsonify({"status": "error", "message": "URL不能为空"}), 400
        
        # 执行预测
        start_pred = time.time()
        result = classifier.predict(url)
        pred_time = time.time() - start_pred
        
        return jsonify({
            "status": "success",
            "result": result,
            "processing_time": f"{pred_time:.3f}秒",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"检测失败: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)