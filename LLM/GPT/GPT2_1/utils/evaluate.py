class Evaluate:
    def __init__(self, dataset, context_size):
        self.dataset = dataset
        self.context_size = context_size
        self.tagger = None

    def predict(self, source_indices, target_indices, outputs):
        # INPUT
        # 辞書の取得
        my_dict = self.dataset.word2index
        # dict_keysオブジェクトをリストに変換
        keys_list = list(my_dict.keys())
        # テキストデータの検証
        text = ""
        for index in source_indices[0]:
            # リストからキーを取得
            key_at_index = keys_list[index]
            text += key_at_index + " "
        print('source', text)

        output_index = outputs[0].argmax().item()
        target_index = target_indices[0]

        # dict_keysオブジェクトをリストに変換
        keys_list = list(my_dict.keys())
        # テキストデータの検証
        # リストからキーを取得
        print('target', keys_list[target_index])
        print('output', keys_list[output_index])
        print()

        return keys_list[target_index], keys_list[output_index]