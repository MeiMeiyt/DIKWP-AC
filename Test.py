import csv
import string
from itertools import count

import jieba
import textstat
from collections import Counter
import os
import  re
# 设置 JAVA_HOME 和 Path 环境变量
java_home = "D:/Java/1.8"  # 根据你的实际路径调整
os.environ['JAVA_HOME'] = java_home
os.environ['Path'] = f"{java_home}/bin;" + os.environ.get('Path', '')

from language_tool_python import LanguageTool

# # 测试语言工具是否正常工作
# tool = LanguageTool('zh-CN')
# text = "这是一个测试句子。"
# matches = tool.check(text)
# print(len(matches))
#
def chinese_readability(text):
    words = list(jieba.cut(text))
    word_count = len(words)
    char_count = len(text)
    # sentence_count = text.count('。') + text.count('！') + text.count('？')
    sentence_count = text.count(',')
    readability_score = 206.835 - 1.015 * (char_count / sentence_count) - 84.6 * (char_count / word_count)
    return readability_score

# 2. 词汇复杂度计算
def lexical_complexity(text):
    words = list(jieba.cut(text))
    word_freq = Counter(words)
    low_freq_words = [word for word, freq in word_freq.items() if freq < 3]
    return len(low_freq_words) / len(words)

# 3. 语法正确性检测
from language_tool_python import LanguageTool
def grammar_check(text):
    tool = LanguageTool('zh-CN')
    matches = tool.check(text)
    return len(matches)

if __name__ == "__main__":
 # with open('D:\project\DIKWP\DIKWP\LegalQA-manual-test.csv', 'r', encoding='utf-8') as file:
 #    reader = csv.reader(file)
 #    for i,row in enumerate(reader):
 #        if i==1:
 #         text="".join(row)
 test1=[]
 path="D:\\project\\Test\\output\\"
 testmodel="unprocess\\"
 test1model="deepseek1\\"
 test2model="deepseek2\\"
 for i in range(1,201):
    fpath=path+testmodel+str(i)+".txt"
    fpath1=path+test1model+str(i)+".txt"
    fpath2=path+test2model+str(i)+".txt"
 #    with open(fpath1 ,'r', encoding='gbk') as f:
 #        string1=f.read()
 #        print("*"*100,i)
 #        position=string1.find('《')
 #        result1=""
 #        if position!=-1:
 #           result1 = string1[:position].strip()
 #           # 综合评估
 #           readability = chinese_readability(result1)
 #           complexity = lexical_complexity(result1)
 #           grammar_errors = grammar_check(result1)
 #
 #           print("可读性分数:", readability)
 #           print("词汇复杂度:", complexity)
 #           print("语法错误数量:", grammar_errors)
 #           # str2=readability+","+complexity+","+grammar_errors
 #           test1.append([readability,complexity,grammar_errors])
 #           print(f"{readability},{complexity},{grammar_errors}")

    with open(fpath2 ,'r') as f:
        string1=f.read()
        print("*"*100,i)
        # position=string1.find('经过3-No问题处理后的资源：')
        # result1=""
        # if position!=-1:
        #    result1 = string1[position:].strip()
           # 综合评估
        print(string1)
        readability = chinese_readability(string1)
        if readability<float(-200):
            print("小于")
            readability = 0
        complexity = lexical_complexity(string1)
        grammar_errors = grammar_check(string1)


        print("可读性分数:", readability)
        print("词汇复杂度:", complexity)
        print("语法错误数量:", grammar_errors)
        # str2=readability+","+complexity+","+grammar_errors
        test1.append([readability, complexity, grammar_errors])
        print(f"{readability},{complexity},{grammar_errors}")



 with open("eval\\tradition_eval_3NoDIKWP.csv","w",encoding="UTF-8",newline="") as f:
    w=csv.writer(f)
    for i in  test1:
           w.writerow(i)

  # text="""问者
  # 相同语义(数据): [“枫叶”牌商标玻璃胶, 哈尔滨振兴装饰材料实业有限公司, 省工商局, 国家商标总局, 哈尔滨市道外区华隆装饰商店]
  # 不同语义(信息): [省工商局扣押“枫叶”牌商标玻璃胶, 国家商标总局认定“枫叶”商标合法, 哈尔滨振兴装饰材料实业有限公司的“楓葉”牌商标注册]
  # 不同语义(智慧): [省工商局的行为可能违法, 竞争对手可能利用工商执法部门进行不正当竞争]
  # 不同语义(意图): [[澄清事实真相, 停止违法行政行为]]
  # 完整语义(绝对/相对): [“枫叶”牌商标玻璃胶的合法注册和使用历史, 哈尔滨振兴装饰材料实业有限公司的“楓葉”牌商标注册情况]
  # 答者
  # 相同语义(数据): [“枫叶”牌商标玻璃胶, 省工商局, 国家商标总局]
  # 不同语义(信息): [省工商局的做法可能违法]
  # 不同语义(智慧): [建议提起商标争议诉讼或侵权诉讼]
  # 不同语义(意图): [[解决商标争议, 维护合法权益]]
  # 完整语义(绝对/相对): [省工商局的行为可能违法, 建议法律途径解决]
  # """
 # print("Flesch Reading Ease:", textstat.flesch_reading_ease(text))
 # print("Flesch-Kincaid Grade Level:", textstat.flesch_kincaid_grade(text))
 # print("Gunning Fog Index:", textstat.gunning_fog(text))
 # print("SMOG Index:", textstat.smog_index(text))
 # print("Automated Readability Index:", textstat.automated_readability_index(text))

