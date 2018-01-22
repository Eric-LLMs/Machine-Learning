import os
from enum import Enum

class rule_col(Enum):
    WritingStyles = 0,  # "体裁"
    Theme = 1,  # "题材"
    TopicDepth = 2,  # "主题深度"
    LanguageStyle = 3,  # "语言风格"
    CharacterNum = 4,  # "字数"
    ImageRatio = 5,  # "图片比例"
    LabelGrade = 6,  # "年级"

def get_rule_col_index(rule_name):
    return rule_col[rule_name].value

#{'index':{'规则名称'：规则值，'规则名称'：规则值，'规则名称'：规则值,'label':年级}}
def get_rules(rule_path):
    dic_rules = {}
    rule_index = 0
    for line in  open(rule_path):
        dic_rule = {}
        rules = line.replace('\n', '').split('\t')
        label = ''
        for rule_pair in rules:
            key = rule_pair.split('\t')[0]
            value = rule_pair.split('\t')[1]
            if key!= 'label':
               dic_rule[key] = value
            if key=='lable':
               label = value
            dic_rule[key]= value
        dic_rules[rule_index]=dic_rule
        dic_rules['label'] = label
    return  dic_rules

#{'index':{'规则名称'：规则值，'规则名称'：规则值，'规则名称'：规则值,'label':年级}}
def labeled_data(data_path,rule_path):
    rules = get_rules(rule_path)
    for line in open(data_path):
       data_item = line.split('\t')
       is_right = -1
       label = ''
       for rule_pairs in rules.items():
           is_right = 1
           if rule_pairs.key == 'lable':
               label = rule_pairs.vaule
               continue
           for rule_pair in rule_pairs.item():
               if data_item[get_rule_col_index(rule_pair.key)] != rule_pair.vaule:
                   is_right = 0
                   break
       if is_right==1:
           # 写入(line,label)
           #.......
           break

if __name__=='__main__':
    pass