import torch
import MeCab, unidic_lite

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
    return pad_mask

class Generate:
    def __init__(self, word2index, index2word, token_size, device):
        self.word2index = word2index
        self.index2word = index2word
        self.token_size = token_size
        self.tagger = MeCab.Tagger(f"-Owakati -d {unidic_lite.DICDIR}")
        self.device = device

    def sequence2indices(self, sequence):
        indices = []
        for word in sequence.split():
            try:
                index = self.word2index[word]
            except:
                index = self.word2index['<UNK>'] # 未登録の単語として処理
            indices.append(index)

        return indices


    def generate(self, corpus, model, max_token_size=100, eos='<EOS>'):
        model.eval()
        # model.cpu()
        pad = self.word2index['<PAD>']
        corpus = self.tagger.parse(corpus)

        source = self.sequence2indices(corpus)
        source = source[:self.token_size]
        # print(source, corpus)

        txt = ""
        for i in range(max_token_size):
            inputs = [source + [pad] * (self.token_size - len(source))]
            inputs = torch.LongTensor(inputs).to(self.device)

            outputs ,_ = model(inputs, None)
            index = torch.argmax(outputs).item()


            #indices.append(index)
            source.append(index)
            source = source[1:]
            
            next_word = self.index2word[index]
            
            txt += next_word + ""
            
            if next_word == eos:
                break
        
        print(f"corpus: {corpus}")
        print(f"Generated text: {txt}")
