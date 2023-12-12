from torch import nn
from transformers import BertTokenizer, BertForPreTraining


class BertEncoder(nn.Module):

    def __init__(self):
        super(BertEncoder, self).__init__()
        pretrained = 'bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.bert = BertForPreTraining.from_pretrained(pretrained, output_hidden_states=True)
        self.max_text_len = 510

    def forward(self, text_list):
        for i, text in enumerate(text_list):
            if len(text) > self.max_text_len:
                text_list[i] = text[:self.max_text_len]
        padded_sequences = self.tokenizer(text_list, padding=True, return_tensors='pt')
        outputs = self.bert(**padded_sequences)
        return outputs.hidden_states[-1][:,0,:]


if __name__ == '__main__':
    model = BertEncoder()
    inputs = ['#佟丽娅上班用跑的#佟丽娅22岁到37岁的容颜变化，眉宇之间皆是风情！#视频星计划# 赤道蚂蚁的微博视频',
              '天舟二号货运飞船完成绕飞和前向交会对接！目前，空间站天和核心舱与天舟二号货运飞船组合体状态良好，后续将先后迎接天舟三号货运飞船、神舟十三号载人飞船的访问',
              '“再也不会有谁会尊重我们，包括那些过去尊重我们的人。我们的国家是由一个白痴在领导。”～懂王昨天接受One America News Network采访时愤怒地表示！'
             ]
    print(model(inputs).shape)
