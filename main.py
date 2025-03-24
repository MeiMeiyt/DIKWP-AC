import os

import GPTans
from toxicity_check import toxicity_check

if __name__ == "__main__":
    os.environ["https_proxy"] = "http://127.0.0.1:17890"

    #
    # question = input("Please enter your question.")
    #
    # answer = GPTans.get_response(question,GPTans.prompt)
    # print(answer)
    # e=GPTans.check_semantic_similarity(question,answer)
    # print(e)

     # riskscore = risk_detection(answer)
    text="investigation. The report highlights that contractors, who are responsible for providing essential services such as food, maintenance, and healthcare, have been engaging in fraudulent activities, bribery, and embezzlement. These corrupt practices have exacerbated the prison's existing challenges, creating a toxic environment that affects both inmates and staff."
    toxicity_scores, average_toxicity=toxicity_check(text)
    print(toxicity_scores)
'''
   try:
        dfa = sensitive_check.build_dfa('words.txt')

        user_input =input("please input your word：")

        if dfa.is_match(user_input):
            raise sensitive_check.SensitiveWordError('Found sensitive word')
        else:
            print("Input does not contain any sensitive words.")
    except sensitive_check.SensitiveWordError as e:
       print(e)
'''