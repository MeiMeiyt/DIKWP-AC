# from docx import Document
#
# import re
# def read_docx(file_path, delimiter='<换页>'):
#     # 打开文档
#     doc = Document(file_path)
#
#     # 初始化一个空字符串用于存储文档中的所有文本
#     full_text = []
#
#     # 遍历文档中的每一个段落
#     for para in doc.paragraphs:
#         # 将每个段落的文本添加到full_text列表中
#         full_text.append(para.text)
#
#     # 将列表转换成字符串，并以指定的分隔符进行分割
#     text_str = '\n'.join(full_text)  # 使用换行符连接段落
#     split_text = text_str.split(delimiter)
#
#     # 返回分割后的文本列表
#     return split_text





def remove_time_stamps(text):
    # 定义正则表达式模式，匹配 (HH:MM:SS) 格式的字符串
    pattern = r'\（?\d{2}:\d{2}:\d{2}\）?'

    # 使用 re.sub() 函数替换匹配到的时间戳为一个空字符串
    cleaned_text = re.sub(pattern, '', text)

    return cleaned_text


def getdata():
# 指定文件路径
    datalist=[]
    file_path = 'C:\\Users\\lenovo\通义千问\\3.docx'
    # 调用函数并打印结果
    split_text = read_docx(file_path)
    index=0
    for i in split_text:
        clean=remove_time_stamps(i)
        index=index+1
        datalist.append(clean)
    print(datalist)
    return datalist






