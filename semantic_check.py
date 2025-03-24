import csv
import os
import re
from datasets import  load_dataset
import openai
from sympy import print_gtk

import GPTans
from GPTans import check_semantic_similarity


def unsafe_control():
    path1 = "D:\\project\\Test\\output\\deepseek1\\"
    path2="D:\\project\\Test\\unsafe_data\\"
    path3="D:\\project\\Test\\output\\toxicity\\"
    #双方交互
    # for i in range(1, 201):
    #  path=path1+str(i)+".txt"
    #  with open(path,'r') as f:
    #     string1=f.read()
    #     print(string1)
    #     print("*"*100,i)
    #     # position=string1.find('《')
    #     # 提取问者的相同语义数据
    #     asker_start = string1.find("问者\n相同语义(数据):") + len("问者\n相同语义(数据):")
    #     asker_end = string1.find("]", asker_start) + 1
    #     asker_data = string1[asker_start:asker_end].strip()
    #     # 提取答者的相同语义数据
    #     answerer_start = string1.find("答者\n相同语义(数据):") + len("答者\n相同语义(数据):")
    #     answerer_end = string1.find("]", answerer_start) + 1
    #     answerer_data = string1[answerer_start:answerer_end].strip()

    for i in range(1, 11):
        path = path3 + str(i) + ".txt"
        with open(path, 'r', encoding="utf-8") as f:
            string1 = f.read()
            print(string1)
            print("*" * 100, i)
            # position=string1.find('《')
            start = string1.find("相同语义(数据):") + len("相同语义(数据):")
            end = string1.find("]", start) + 1
            data = string1[start:end].strip()
        info_start = string1.find("不同语义(信息):") + len("不同语义(信息):")
        info_end = string1.find("]", info_start) + 1
        info= string1[info_start:info_end].strip()
        wis_start = string1.find("不同语义(智慧):") + len("不同语义(智慧):")
        wis_end = string1.find("]", wis_start) + 1
        wis= string1[wis_start: wis_end].strip()
        pup_start = string1.find("不同语义(意图):") + len("不同语义(意图):")
        pup_end = string1.find("]", pup_start) + 1
        pup= string1[pup_start:pup_end].strip()

        kng_start = string1.find("完整语义(绝对/相对):") + len("完整语义(绝对/相对):")
        kng_end = string1.find("]", kng_start) + 1
        kng = string1[kng_start:kng_end].strip()

        result1=""
        # 双方交互
        # path3=path2+"bias.txt"
        # with open(path3, 'r',encoding="utf-8") as file:
        #  for k,line in enumerate(file):
        #       print(line)
        #       print("k:"+str(k)+"i:"+str(i)+"****************************")
        #       if GPTans.check_semantic_similarity(asker_data,line)==True:
        #         print(str(k)+"*********相同语义"+str(GPTans.check_semantic_similarity(asker_data,line)))
        #         test_same_asker.append("case:"+str(i)+"class:"+str(k+1))
        #       if GPTans.check_semantic_similarity(asker_data,line)==True:
        #         print(str(k)+"*********相同语义"+str(GPTans.check_semantic_similarity(answerer_data,line)))
        #         test_same_answer.append("case:"+str(i)+"class:"+str(k+1))

        path4 = path2 + "illegally_En.txt"
        with open(path4, 'r', encoding="utf-8") as file:
            for k, line in enumerate(file):
                print(line)
                print("k:" + str(k) + "i:" + str(i) + "****************************")
                if GPTans.check_semantic_similarity(data, line) == True:
                    print(str(k) + "*********相同语义" + str(GPTans.check_semantic_similarity(data, line)))
                    test_same_asker.append("case:" + str(i) + "class:" + str(k + 1))
                if GPTans.check_semantic_similarity(info, line) == True:
                        print(str(k) + "*********不同语义(信息)" + str(GPTans.check_semantic_similarity(info, line)))
                        test_different_info.append("case:" + str(i) + "class:" + str(k + 1))
                if GPTans.check_semantic_similarity(wis, line) == True:
                    print(str(k) + "*********不同语义(智慧)" + str(GPTans.check_semantic_similarity(wis, line)))
                    test_different_wisdom.append("case:" + str(i) + "class:" + str(k + 1))
                if GPTans.check_semantic_similarity(pup, line) == True:
                        print(str(k) + "*********不同语义(意图)" + str(GPTans.check_semantic_similarity(pup, line)))
                        test_different_purpose.append("case:" + str(i) + "class:" + str(k + 1))
                if GPTans.check_semantic_similarity(kng, line) == True:
                    print(str(k) + "*********完整语义(绝对/相对):" + str(GPTans.check_semantic_similarity(pup, line)))
                    test_complete.append("case:" + str(i) + "class:" + str(k + 1))

if __name__ == "__main__":
    os.environ["https_proxy"] = "http://127.0.0.1:17890"
    test_same_asker = []
    test_same_answer = []
    test_different_info=[]
    test_different_wisdom=[]
    test_different_purpose=[]
    test_complete=[]
    #
    # unsafe_control()
    # with open("unsafe_data\\bias_check_data_asker.csv","w",encoding="UTF-8",newline="") as f1:
    #  w=csv.writer(f1)
    #  for j in  test_same_asker:
    #      w.writerow(j)
    # with open("unsafe_data\\bias_check_data_answer.csv","w",encoding="UTF-8",newline="") as f2:
    #  m=csv.writer(f2)
    #  for j in  test_same_answer:
    #      m.writerow(j)
    # with open("unsafe_data\\bias_check_info.csv","w",encoding="UTF-8",newline="") as f2:
    #  m=csv.writer(f2)
    #  for i in  test_different_info:
    #      m.writerow(i)
    # with open("unsafe_data\\bias_check_wisdom.csv","w",encoding="UTF-8",newline="") as f3:
    #  n=csv.writer(f3)
    #  for k in  test_different_wisdom:
    #      n.writerow(k)
    # with open("unsafe_data\\bias_check_purpose.csv","w",encoding="UTF-8",newline="") as f4:
    #  l=csv.writer(f4)
    #  for i in  test_different_purpose:
    #      l.writerow(i)
    # with open("unsafe_data\\bias_check_knowledge.csv","w",encoding="UTF-8",newline="") as f5:
    #  L=csv.writer(f5)
    #  for i in  test_complete:
    #      L.writerow(i)
    # 写100条数据到txt中，将real-toxicity-prompts数据集
    # output_file_path = "toxicity_prompts_text.txt"
    # ds = load_dataset("allenai/real-toxicity-prompts")
    # with open(output_file_path, "w", encoding="utf-8") as file:
    #  for example in ds['train'].select(range(100)):  # 假设数据集包含一个名为'train'的split
    #     prompt_text = example['prompt']['text']
    #     file.write(prompt_text + "\n")


    unsafe_control()
    with open("unsafe_data\\illegal\\illegal_check_toxicity_data.csv","w",encoding="UTF-8",newline="") as f1:
     w=csv.writer(f1)
     for j in  test_same_asker:
         w.writerow(j)
    with open("unsafe_data\\illegal\\illegal_check_toxicity_info.csv","w",encoding="UTF-8",newline="") as f2:
     w=csv.writer(f2)
     for j in  test_different_info:
         w.writerow(j)
    with open("unsafe_data\\illegal\\illegal_check_toxicity_wis.csv","w",encoding="UTF-8",newline="") as f3:
     w=csv.writer(f3)
     for j in  test_different_wisdom:
         w.writerow(j)
    with open("unsafe_data\\illegal\\illegal_check_toxicity_pup.csv","w",encoding="UTF-8",newline="") as f4:
     w=csv.writer(f4)
     for j in  test_different_purpose:
         w.writerow(j)
    with open("unsafe_data\illegal\\illegal_check_toxicity_kng.csv","w",encoding="UTF-8",newline="") as f5:
     w=csv.writer(f5)
     for j in  test_complete:
         w.writerow(j)

