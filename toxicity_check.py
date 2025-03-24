# 初始化PerspectiveAPI实例
import json
import os

import numpy as np
from googleapiclient import discovery
from detoxify import Detoxify

def toxicity_check(text):
    os.environ["https_proxy"] = "http://127.0.0.1:17890"
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    # API_KEY = '
    # client = discovery.build(
    #     "commentanalyzer",
    #     "v1alpha1",
    #     developerKey=API_KEY,
    #     discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    #     static_discovery=False,
    #
    # )
    # analyze_request = {
    #     'comment': {'text': 'friendly greetings from python'},
    #     'requestedAttributes': {'TOXICITY': {}}
    # }
    # response = client.comments().analyze(body=analyze_request).execute()
    # print(json.dumps(response, indent=2))


    # 加载模型（支持CPU/GPU）
    model = Detoxify('original')  # 可选模型：'original', 'unbiased', 'multilingual'

    # 批量预测
    results = model.predict(text)

    # 提取"toxicity"分数并计算平均值
    toxicity_scores = results['toxicity']
    average_toxicity = np.mean(toxicity_scores)

    print(f"各文本毒性评分: {toxicity_scores}")
    print(f"平均毒性评分: {average_toxicity:.4f}")
    # results = Detoxify('original').predict("Your text her
    #e.")
    # print(results)  # 输出各标签的概率
    return  toxicity_scores, average_toxicity


