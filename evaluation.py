import csv
import json
import os
import re
import time
from doctest import testmod
from http.client import responses

import openai
from openai import OpenAI, api_key

from clean import getdata1, extract_numbers
from getallfile import getdata
import requests

def eval():
 path="D:\\project\\Test\\output\\"
 testmodel="unprocess\\"
 test1model="deepseek1\\"
 test2model="deepseek2\\"
#  datalist=getdata()

 test1=[]
 test2=[]
 test3=[]
 for i in range(1,201):
    fpath=path+testmodel+str(i)+".txt"
    fpath1=path+test1model+str(i)+".txt"
    fpath2 = path + test2model + str(i) + ".txt"
    sep=""
    prpmote=("你是一个公正的语言、逻辑评价官，以下有两个文本，model1是未处理的文本，model2、model3是处理后的文本，请你从清晰度、实用性、完整度，来评价一下model1，model2和model3文本答的质量，每个指标分数在1-100之间可以是任意的两位小数，"
             "其中score1为清晰度，score2为实用性，score3为完整度，清晰度：指处理后的文本是否能够以简洁明了的方式传达核心信息，使读者无需额外背景知识就能理解主要内容。实用性：指处理后的文本是否提供了切实可行的建议或解决方案，帮助解决实际问。完整度：指处理后的文本是否全面涵盖了原始文本中的所有重要信息，没有遗漏关键细节。请务必按以下格式返回，不要解释只要分数："
             "model1[score1,score2,score3]  model2[score1,score2,score3]  model3[score1,score2,score3]")

    with open(fpath1 ,'r', encoding='gbk') as f:
        string1=f.read()
        print("*"*100,i)
        position=string1.find('《')
        result1=""
        if position!=-1:
           result1 = string1[:position].strip()
         # else:
         #   print("未找到《")
        # 使用 join 方法将剩余的部分拼接成新的字符串




    # with open(fpath1 ,'r') as f:
    #     string2=f.read()
    # print("*"*100,i)

    with open(fpath, 'r', encoding='utf-8') as file:
        string2=file.read()
        # matchs= re.findall(r'$(.*?)$', string)
        # for match in matchs:
        #      string2=match
        #      print(match)
    with open(fpath2, 'r') as file:
        string3=file.read()
    fstring=prpmote+"<model1>"+string2+"<model2>"+result1+"<model3>"+string3
    print(fstring)

    # os.environ["https_proxy"] = "http://127.0.0.1:17890"
    client = OpenAI(
   ,
        base_url="",
    )
    completion = client.chat.completions.create(
        # model="qwen-plus",
        model="deepseek-reasoner",
        # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': fstring}],
    )
    response = completion.model_dump()
    outputstring = response["choices"][0]['message']['content']

    print(outputstring)
    time.sleep(1)
    # print("Status Code", response.status_code)
    # print("JSON Response ", response)
    # outputstring=response.json()["choices"][0]['message']['content']
    eval1, eval2, eval3 = getdata1(outputstring)
    test1.append(eval1)
    test2.append(eval2)
    test3.append(eval3)
    # url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    #
    # headers = {
    #     'Content-Type': 'application/json',
    #     'Authorization': ''
    #     # <-- 把 fkxxxxx 替换成你自己的 Forward Key，注意前面的 Bearer 要保留，并且和 Key 中间有一个空格。
    # }
    #
    # data = {
    #     "model": "qwen-omni-turbo",
    #     "messages": [{"role": "user", "content": fstring}]
    # }
    #                 print(fstring)
    #
    # response = requests.post(url, headers=headers, json=data)

 with open("eval\deepseekR1_eval.csv","w",encoding="UTF-8",newline="") as f:
    w=csv.writer(f)
    for i in  test1:
         w.writerow(i)
#
 with open("eval\deepseekR1_eval_DIKWP.csv","w",encoding="UTF-8",newline="") as f1:
     w=csv.writer(f1)
     for j in  test2:
         w.writerow(j)

 with open("eval\deepseekR1_eval_3NoDIKWP.csv", "w", encoding="UTF-8", newline="") as f2:
     w = csv.writer(f2)
     for k in test3:
         w.writerow(k)