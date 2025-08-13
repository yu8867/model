import torch

def create_pad_mask(source):
    # True -> 1; False -> 0
    source = source.ne(0) * 1
    sq_masks = []
    for tokens in source:
        rows = []
        for n in range(1, len(tokens) + 1):
            mask = torch.cat((tokens[:n], tokens[n:] * 0), dim=0)
            rows.append(mask)
        sq_mask = torch.stack(rows)
        sq_masks.append(sq_mask)

    pad_mask = torch.stack(sq_masks).unsqueeze(1)

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


    def generate(self, corpus, model, mask=None, max_token_size=500, eos='<EOS>'):
        model.eval()
        model.cpu()
        pad = self.dataset.word2index['<PAD>']

        source = self.dataset.sequence2indices(corpus)
        source = source[:self.context_size]
        #indices = copy.copy(source)

        for i in range(max_token_size):
            inputs = [source + [pad] * (self.context_size - len(source))]
            inputs = torch.LongTensor(inputs).cpu()
            
            if mask is not None:
                mask = create_pad_mask(inputs)

            outputs ,_ = model(inputs, mask)
            if 1:
                index = torch.argmax(outputs).item()
            else:
                k = 3
                opk_values, topk_indices = torch.topk(outputs, k)
                # k個の最大値からランダムに1つをサンプリング
                topk_index = torch.randint(0, topk_indices.size(1), (1,))
                index = topk_indices[0, topk_index.item()].tolist()


            #indices.append(index)
            source.append(index)
            source = source[1:]
            
            next_word = self.dataset.index2word[index]

            print(next_word ,end="")
            
            if next_word == eos:
                break