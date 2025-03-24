# 假设这是你的输入字符串
input_str = "model1[8.5, 9.0, 8.0]  model2[9.0, 8.5, 8.5] model3[9.4, 8.5, 8.2]"
def getdata1(input_str):
    model1_data = input_str.split('model1[')[1].split(']')[0]
    model2_data = input_str.split('model2[')[1].split(']')[0]
    model3_data = input_str.split('model3[')[1].split(']')[0]

    # 提取数字
    model1_numbers = extract_numbers(model1_data)
    model2_numbers = extract_numbers(model2_data)
    model3_numbers = extract_numbers(model3_data)
    print("Model1 numbers:", model1_numbers)
    print("Model2 numbers:", model2_numbers)
    print("Model3 numbers:", model3_numbers)
    return model1_numbers,model2_numbers,model3_numbers
# 定义一个函数来提取模型内的数字
def extract_numbers(model_str):
    # 去掉方括号
    model_str = model_str.strip('[]')
    # 分割字符串并转换为浮点数
    return [float(num) for num in model_str.split(',')]

# 分割输入字符串以提取每个模型的数据
getdata1(input_str)