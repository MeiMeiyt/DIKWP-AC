import os
import time
import evaluation
from  openai import OpenAI
import csv
# from getallfile import read_docx
DIKWPtext="""
句子包含两个角色，问者和答者，分别进行语义空间映射，语义空间包括相同语义，不同语义，以及完整语义，组成由数据、信息、知识、智慧以及意图五个元素构成，由此构建知识图谱，以下是相同语义、不同语义以及完整语义的定义，按照定义将文本映射过去，按照输出格式输出结果，只保留输出格式内容。
相同语义
定义：所有文本内容都是数据，共享同一概念的数据。
示例：如“羊群”中的每只羊或“苹果”的不同实例，尽管个体特征各异，但都归类为同一个概念。
不同语义
定义：涉及信息、智慧和意图的不同层面。
信息：具体事实或不同认知下体现差异。
智慧：从信息中提炼出的高级理解或推断。
意图：二元组形式，输入（问题或现象）与输出（目标或期望结果）之间的关系。
完整语义
定义：构成知识的完整信息。
相对完整：完成特定任务所需的所有信息。
绝对完整：不仅包含完成任务所需的信息，还包括相关扩展知识。

输出格式：
问者/答者
相同语义(数据): [数据1,数据2， ...]
...
不同语义(信息):[信息1,信息2， ...]
不同语义(智慧): [智慧1, 智慧2, ...]
不同语义(意图): [[意图1, 意图2, ...]
...
完整语义(绝对/相对): [知识1, 知识2, ...]
...
《分隔符》  在对所有轮对话完成分析后，判断是问者和答者否存在模糊、不清晰的地方，利用数据 信息、知识、智慧去解释意图的变化和模糊、不清晰的地方及其变化 
输出格式如下：模糊、不清晰的地方：
变化原因：
"""
text1="""
句子包含两个角色，问者和答者，对问者和答者，进行如下操作：
1.分别进行语义空间映射，语义空间包括相同语义，不同语义，以及完整语义，组成由数据、信息、知识、智慧以及意图五个元素构成，
2.找出双方存在的模糊、不清晰的地方，具体为为3-No问题，包括不一致，不完整，不精确。
3.对双方存在3-No问题的资源进行转换，给出转换处理后的问者和答者完整内容，非建议，请句子完整，逻辑清晰。
以下是相关定义：
相同语义
定义：所有文本内容都是数据，共享同一概念的数据。
示例：如“羊群”中的每只羊或“苹果”的不同实例，尽管个体特征各异，但都归类为同一个概念。
不同语义
定义：涉及信息、智慧和意图的不同层面。
信息：具体事实或不同认知下体现差异。
智慧：从信息中提炼出的高级理解或推断。
意图：二元组形式，输入（问题或现象）与输出（目标或期望结果）之间的关系。
完整语义
定义：构成知识的完整信息。
相对完整：完成特定任务所需的所有信息。
绝对完整：不仅包含完成任务所需的信息，还包括相关扩展知识。

三类定义如下：
1.不完整性 (Incomplete)
设I是输入文本，F是所有可能的事实集合，而K是从文本中提取出的事实子集。
公式：如果存在f属于F但f不属于K，即K不包含F，则I是不完整的。

2. 不一致性 (Inconsistent)
设S是文本中的陈述集合，Consistent(si,sj) 是一个布尔函数，用于检查两个陈述si和sj是否一致。
公式：如果存在 si,sj∈S使得Consistent(si,sj)=false，则I 是不一致的。

3. 不精确性 (Imprecise)
设P是对某个事件或数值的概率分布函数，E[P] 是其期望值，σ(P) 是标准差。
公式：对于给定的事件 A，如果 P(A) 的范围过大或者σ(P)过大，则I是不精确的。

请务必按如下输出格式输出：
不一致：
相同语义(数据): [数据1,数据2， ...]
...
不同语义(信息):[信息1,信息2， ...]
不同语义(智慧): [智慧1, 智慧2, ...]
不同语义(意图): [[意图1, 意图2, ...]
...
完整语义(知识): [知识1, 知识2, ...]
...
不完整：
相同语义(数据): [数据1,数据2， ...]
...
不同语义(信息):[信息1,信息2， ...]
不同语义(智慧): [智慧1, 智慧2, ...]
不同语义(意图): [[意图1, 意图2, ...]
...
完整语义(知识): [知识1, 知识2, ...]
...
不精确：
相同语义(数据): [数据1,数据2， ...]
...
不同语义(信息):[信息1,信息2， ...]
不同语义(智慧): [智慧1, 智慧2, ...]
不同语义(意图): [[意图1, 意图2, ...]
...
完整语义(知识): [知识1, 知识2, ...]
...
《分隔符》
经过3-No问题处理后的资源：
问者/答者：
相同语义(数据): [数据1,数据2， ...]
...
不同语义(信息):[信息1,信息2， ...]
不同语义(智慧): [智慧1, 智慧2, ...]
不同语义(意图): [[意图1, 意图2, ...]
...
完整语义(绝对/相对): [知识1, 知识2, ...]
...
"""

text2=("""找出上文双方表述存在的模糊、不清晰的地方，具体为为3-No问题，包括不一致，不完整，不精确。请分别对问者和答者存在的3-No问题进行转化、替换处理，请仅给出处理后的内容，3-No问题定义如下：
1.不完整性 (Incomplete)
设I是输入文本，F是所有可能的事实集合，而K是从文本中提取出的事实子集。
公式：如果存在f属于F但f不属于K，即K不包含F，则I是不完整的。

2. 不一致性 (Inconsistent)
设S是文本中的陈述集合，Consistent(si,sj) 是一个布尔函数，用于检查两个陈述si和sj是否一致。
公式：如果存在 si,sj∈S使得Consistent(si,sj)=false，则I 是不一致的。

3. 不精确性 (Imprecise)
设P是对某个事件或数值的概率分布函数，E[P] 是其期望值，σ(P) 是标准差。
公式：对于给定的事件 A，如果 P(A) 的范围过大或者σ(P)过大，则I是不精确的。

请按以下格式输出：
问者/答者
相同语义(数据): [数据1,数据2， ...]
...
不同语义(信息):[信息1,信息2， ...]
不同语义(智慧): [智慧1, 智慧2, ...]
不同语义(意图): [[意图1, 意图2, ...]
...
完整语义(绝对/相对): [知识1, 知识2, ...]
... 
       """)

#
client = OpenAI(api_key="", base_url="")
#
#
def getresponse(text,index,client,flag):
    if flag==0:
        path="D:\\project\\Test\\output\\deepseek1\\" + str(index) + ".txt"
    if flag==1:
        path = "D:\\project\\Test\\output\\deepseek2\\" + str(index) + ".txt"
    try:
        print("start")
        print(text)
        completion = client.chat.completions.create(
            model="deepseek-chat",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': text}
                ]
            ,stream=True
        )
        full_content = ""
        print("流式输出内容为：")
        for chunk in completion:
            # print(chunk.choices[0].delta.content)
            full_content += chunk.choices[0].delta.content
        print(f"完整内容为：{full_content}")
        with open(path,"w") as f:
            absolute_path=os.path.abspath(path)
            print(f"Writing to: {absolute_path}")
            f.write(full_content)
    except Exception as e:
        print(f"错误信息：{e}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")

def get3No(text,index):
    path = "D:\\project\\Test\\output\\deepseek2\\" + str(index) + ".txt"
    try:
         print("start")
         print(text)
         completion = client.chat.completions.create(
             model="deepseek-chat",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
             messages=[
                 {'role': 'system', 'content': 'You are a helpful assistant.'},
                 {'role': 'user', 'content': text}
             ]
             , stream=True
         )
         full_content = ""
         print("流式输出内容为：")
         for chunk in completion:
             # print(chunk.choices[0].delta.content)
             full_content += chunk.choices[0].delta.content
         print(f"完整内容为：{full_content}")
         with open(path, "w") as f:
             absolute_path = os.path.abspath(path)
             print(f"Writing to: {absolute_path}")
             f.write(full_content)
    except Exception as e:
         print(f"错误信息：{e}")
         print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
# datalist=read_docx("3.docx")
# client = OpenAI(
#     OpenAI(api_key="sk-73c107ed0477410d82f656fbcf3d0ac5", base_url="https://api.deepseek.com")
# )
#


if __name__ == "__main__":
 # with open('D:\project\Test\LegalQA-manual-test.csv', 'r', encoding='utf-8') as file:
 #    reader = csv.reader(file)
 #    index =0
 #    flag=1
 #    for j,i in enumerate(reader):
 #        print(f"Index: {j}")
 #        if j == 5:
 #            break
 #        s = "".join(i)
 #        textRes = s + text1
 #        getresponse(textRes, index, client, flag)
 #        time.sleep(1)
 #        print("成功")
 #        index = index + 1
 # test1 = []
 # path = "D:\\project\\Test\\output\\"
 # for i in range(86, 201):
 #         fpath = path + "deepseek1\\" + str(i) + ".txt"
 #         print("*" * 100 + str(i))
 #         with open(fpath, 'r') as file:
 #             reader = file.read()
 #             position = reader.find('《')
 #             result1 = ""
 #             if position != -1:
 #                 result1 = reader[:position].strip()
 #             textRes = result1 + text2
 #             get3No(textRes, i)
 #             time.sleep(1)
 #             print("成功")

 # index=0
 # flag=1

     evaluation.eval()

    #
    # with open('D:\project\Test\LegalQA-manual-test.csv', 'r', encoding='utf-8') as file:
    #     reader = csv.reader(file)
    #     for j,row in enumerate(reader):
    #        print(j)
    #        print(row)
    #        path1 = "D:\\project\\Test\\output\\unprocess\\" + str(j) + ".txt"
    #        with open(path1, "w", encoding='utf-8') as f:
    #             f.write(str(row))


