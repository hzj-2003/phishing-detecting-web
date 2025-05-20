import tensorflow as tf
import numpy as np
import pickle
import re
import os
from collections import Counter

# 配置路径（请确认路径正确）
BASE_DIR = r"C:\Users\Administrator\Desktop\URLNet"
MODEL_DIR = os.path.join(BASE_DIR, "urlnet_modelcheckpoints")
VOCAB_SAVE_PATH = os.path.join(MODEL_DIR, "vocab.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "model-715")

class URLNetClassifier:
    def __init__(self):
        # 生成基础词汇表（如果不存在）
        if not os.path.exists(VOCAB_SAVE_PATH):
            self._generate_base_vocab()
            
        with open(VOCAB_SAVE_PATH, 'rb') as f:
            self.vocab = pickle.load(f)
            
        # 初始化TensorFlow会话
        self.sess = tf.Session()
        
        try:
            # 加载计算图
            saver = tf.train.import_meta_graph(MODEL_PATH + ".meta")
            saver.restore(self.sess, MODEL_PATH)
            
            # 获取所有必要的张量引用
            graph = self.sess.graph
            self.input_x = graph.get_tensor_by_name("input_x_char_seq:0")
            self.dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")  # 新增关键参数
            self.predictions = graph.get_tensor_by_name("output/predictions:0")
            
            print("[调试] 成功加载以下张量：")
            print(f"Input tensor: {self.input_x.name}")
            print(f"Dropout keep prob: {self.dropout_keep_prob.name}")
            print(f"Output tensor: {self.predictions.name}")
            
        except Exception as e:
            print("[错误] 计算图关键操作列表：")
            for op in graph.get_operations()[-20:]:  # 打印最后20个操作
                print(op.name)
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def _generate_base_vocab(self):
        """生成包含基础字符的词汇表"""
        base_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~:/?#[]@!$&'()*+,;=%")
        vocab = {'<PAD>': 0, '<UNK>': 1}
        vocab.update({c: i+2 for i, c in enumerate(base_chars)})
        
        with open(VOCAB_SAVE_PATH, 'wb') as f:
            pickle.dump(vocab, f)

    def preprocess(self, url):
        """预处理流程（严格匹配训练设置）"""
        # 1. 清理URL
        url = re.sub(r"https?://", "", url).split('/')[0]
        
        # 2. 字符级分词
        tokens = list(url)[:200]
        
        # 3. 转换为索引
        indices = [self.vocab.get(c, 1) for c in tokens]
        
        # 4. 填充处理
        padded = indices + [0]*(200 - len(indices))
        return np.array([padded])  # 保持batch维度

    def predict(self, url):
        input_data = self.preprocess(url)
        try:
            # 关键修改：添加dropout_keep_prob参数
            feed_dict = {
                self.input_x: input_data,
                self.dropout_keep_prob: 1.0  # 推理时关闭dropout
            }
            pred = self.sess.run(self.predictions, feed_dict)
            return "恶意" if pred[0] == 1 else "正常"
        except Exception as e:
            print(f"预测错误细节：{str(e)}")
            return "未知"

if __name__ == "__main__":
    try:
        classifier = URLNetClassifier()
        print("\n--- 测试预测 ---")
        
        test_cases = [
            ("http://google.com", "正常"),
            ("http://malware.com/bad.php?id=<script>", "恶意"),
            ("http://123.45.67.89:8080", "恶意"),
            ("https://bank.com/login.php", "正常")
        ]
        
        for url, expected in test_cases:
            result = classifier.predict(url)
            status = "✓" if result == expected else "✗"
            print(f"{status} URL: {url[:35].ljust(35)} 预测: {result.ljust(5)} (预期: {expected})")
            
    except Exception as e:
        print(f"致命错误: {str(e)}")
        print("排查建议：")
        print("1. 确认TensorFlow版本为1.13.1")
        print("2. 检查模型文件完整性（应有.meta, .data, .index文件）")
        print("3. 运行以下命令验证环境：")
        print("   python -c 'import tensorflow as tf; print(tf.__version__)'")
