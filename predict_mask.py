from dataclasses import make_dataclass
import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
import numpy as np
from pyknp import Juman

class PredictMaskToken:
    def __init__(self, texts=["このクラスは*でマスクされた単語を予測します。"], predict_num = 5, lang = "ja"): #複数文に未対応
        self.texts = texts
        file_name = str()
        if lang == "ja":
            file_name = "Japanese_L-12_H-768_A-12_E-30_BPE"
        elif lang == "en":
            #file_name = "uncased_L-12_H-768_A-12" #英語学習モデル
            file_name = "bert-base-uncased"

        config = BertConfig.from_json_file('./model/' + file_name + '/bert_config.json')
        model = BertForMaskedLM.from_pretrained('./model/' + file_name + '/pytorch_model.bin', config=config)
        bert_tokenizer = BertTokenizer('./model/' + file_name + '/vocab.txt', do_lower_case=False, do_basic_tokenize=False)

        #juman使って品詞分解する
        tokenized_text_list = [self.hinshi_bunkai(row) for row in texts]

        #先頭に[CLS]、文の間と終わりに[SEP]トークンを入れる作業
        tokenized_text_list[0].insert(0, '[CLS]') #先頭に[CLS]トークン
        tokenized_text = self.insert_sep_token(tokenized_text_list)
        # * を[MASK]に置き換える
        masked_index = self.masked_position(tokenized_text)
        tokenized_text[masked_index] = '[MASK]'
        
        #BERTで処理できるように、辞書を用いてID変換
        tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([tokens])
        #print(tokens_tensor) #これがBERTの入力データ

        #[MASK]を予測し、尤度の高いk単語
        model.eval()
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]
        _,predicted_indexes = torch.topk(predictions[0, masked_index], k=predict_num)
        self.predicted_tokens = bert_tokenizer.convert_ids_to_tokens(predicted_indexes.tolist()) #予測結果

    def show_predict(self): #予測結果の標準出力
        print("入力データ：{}".format(self.texts))
        print("予測結果：{}".format(self.predicted_tokens))
    
    def masked_position(self, tokenized_text): # *のインデックスを返す
        masked_index = tokenized_text.index("*")
        return masked_index

    def insert_sep_token(self, tokenized_text_list): #[SEP]トークンを入れる
        masked_sep_text = list()
        for text in tokenized_text_list:
            text.append('[SEP]')
            masked_sep_text += text
        return masked_sep_text

    def check_ids(self, word="乗り物"): #wordのid検索
        bert_tokenizer = BertTokenizer('./model/Japanese_L-12_H-768_A-12_E-30_BPE/vocab.txt', do_lower_case=False, do_basic_tokenize=False)
        tokens = bert_tokenizer.convert_tokens_to_ids([word])
        tokens_tensor = torch.tensor([tokens])
        print(tokens_tensor)

    def hinshi_bunkai(self, text): #Jumanで品詞分解する
        jumanpp = Juman(jumanpp=False)
        result = jumanpp.analysis(text)
        tokenized_text = [mrph.midasi for mrph in result.mrph_list()]
        #return tokenized_text
        return [row for row in tokenized_text if row != "\\ "]

if __name__ == "__main__":
    text = ["琵琶湖は*県にあります。"]
    #text = ["ここは*です。", "勉強できません。"]
    #text = ["私の主張は単なる*ではなく、確たる証拠に基づいている。"]
    #text = ["立命館は、人類の未来を切り拓くために、学問研究の自由に基づき普遍的な価値の創造と人類的諸課題の解明に邁進する。", "その教育にあたっては、建学の精神と教学理念に基づき、「未来を信じ、未来に生きる」の精神をもって、確かな学力の上に、豊かな個性を花開かせ、正義と倫理をもった地球市民として活躍できる人間の育成に努める。", "立命館は、この憲章の本旨を踏まえ、教育・研究機関として世界と日本の平和的・民主的・持続的発展に貢献する。"]
    #text = ["伝教大師・最澄が開いた天台宗の総本山「*」。", "東塔、西塔、横川の三塔十六谷からなり、数百のお堂や伽藍が点在しています。"]
    #text = ["遅刻(水時計)を設置した天智天皇が、初めて国民に時を知らせたとして「時の記念日」に制定されている6月10日には、全国の*関係者が集う「漏刻祭（ろうこくさい）」が開催されます。", "境内の一角には、国内外の時計や近江神宮所蔵の宝物を展示する「近江神宮時計館宝物館」もあります。"]
    #text = ["pig are * animals."]
    #text = ["Hello I'm a * model."]
    #text = ["* were the hereditary military nobility and officer caste of medieval and early-modern Japan from the late 12th century to their abolition in 1876."]
    
    pmt = PredictMaskToken(text, 10, "ja") #日本語=ja、英語=en
    pmt.show_predict()