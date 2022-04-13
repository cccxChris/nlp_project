from data_pro import clean_str
import pandas as pd

# positive_data_file = './data/emotion_positive.txt'
# positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
# positive_examples = [s.strip() for s in positive_examples]
# x_text = positive_examples
#
# with open('./data/emotion_positive_clean.txt', 'w', encoding='utf-8') as f:
#     for sent in x_text:
#         x_text = clean_str(sent)
#         if len(x_text.split(' ')) <= 2:
#             pass
#         elif len(x_text.split(' ')) >= 100:
#             pass
#         else:
#             f.write(x_text + '\n')
#
# f.close()

# neg_data_file = './data/emotion_negative.txt'
# neg_examples = list(open(neg_data_file, "r", encoding='utf-8').readlines())
# neg_examples = [s.strip() for s in neg_examples]
# x_text = neg_examples

df = pd.read_excel('./data/depression_dataset.xlsx')
df = df.drop_duplicates()

i = 0
text_list = []
with open('data/emotion_negative_test.txt', 'w', encoding='utf-8') as f:
    for index, row in df.iterrows():
        text = row['Text_data']
        label = row['Simple Label']
        x_text = clean_str(text)
        x_text = x_text.rstrip('removed')
        if len(x_text.split(' ')) <= 3:
            pass
        elif len(x_text.split(' ')) >= 150:
            pass
        else:
            if i >= 2800:
                break
            if x_text in text_list:
                continue
            elif label != 'not depression':
                text_list.append(x_text)
                i += 1
                if i >= 2000:
                    f.write(x_text + '\n')
            # if label == 'not depression':
            #     if x_text in text_list:
            #         continue
            #     else:
            #         f.write(x_text + '\n')
            #         text_list.append(x_text)
            #         i += 1

f.close()

# df = pd.read_excel('./data/others_clean.xlsx')
# df = df.drop_duplicates()
#
# i = 0
# text_list = []
# with open('data/emotion_positive_test.txt', 'w', encoding='utf-8') as f:
#     for index, row in df.iterrows():
#         text = row['content']
#         x_text = clean_str(text)
#         x_text = x_text.rstrip('removed')
#         if len(x_text.split(' ')) <= 3:
#             pass
#         elif len(x_text.split(' ')) >= 150:
#             pass
#         else:
#             if i >= 3600:
#                 break
#             if x_text in text_list:
#                 continue                i += 1

#             else:
#                 text_list.append(x_text)
#                 i += 1
#                 if i < 2800:
#                     pass
#                 else:
#                     f.write(x_text + '\n')
#
# f.close()