from torch.utils.data import Dataset
import MeCab, unidic_lite
import torch

class JpTextDataset(Dataset):
    def __init__(self, corpus, max_sequence_length=15):
        self.tagger = MeCab.Tagger(f"-Owakati -d {unidic_lite.DICDIR}")
        self.max_sequence_length = max_sequence_length
        self.vocab = self.get_vocab(corpus)
        self.word2index = {word: idx for idx, word in enumerate(self.vocab)}
        self.index2word = {idx:token for idx, token in enumerate(self.vocab)}
        self._add_padding_word()
        self.tokenized_corpus = self._tokenize_corpus(corpus)

    def _add_padding_word(self):
        self.vocab.append('<PAD>')
        index_pad = len(self.vocab) - 1
        self.word2index['<PAD>'] = index_pad
        self.index2word[index_pad] = ''

    def __len__(self):
        return len(self.tokenized_corpus)

    def __getitem__(self, idx):
        source_indices  = self.tokenized_corpus[idx]
        source_indices = torch.LongTensor(source_indices)
        return {
            'source': source_indices[:self.max_sequence_length],
            'target': source_indices[self.max_sequence_length]
        }

    def _tokenize_corpus(self, corpus):
        """
        全てのコーパスを指定の長さで区切って、行単位に格納します。
        """
        tokenized_corpus = []
        corpus = corpus.split() #['今日', 'は', '晴れ', 'です', '。']
        tokenized_line = []
        sequence_size = self.max_sequence_length + 1

        for i in range(len(corpus) - sequence_size):
            sequence = corpus[i:i + sequence_size] #['は', '晴れ', 'です']

            for token in sequence:
                index = self.word2index[token]
                tokenized_line.append(index)

                if len(tokenized_line) == sequence_size:
                    tokenized_corpus.append(tokenized_line)
                    tokenized_line = []

        return tokenized_corpus

    def get_vocab(self, corpus):
        """
        コーパスからsetコマンドで、語彙を一意に格納します。
        語彙順にソートして返却します。 
        """
        vocab = set(token for token in corpus.split())
        return sorted(vocab)

    def _read_corpus(self, corpus_path):
        with open(corpus_path, 'rb') as f:
            lines = []
            for line in f:
                line = line.strip().lower().decode("ascii", "ignore")
                if len(line) == 0:
                    continue
                lines.append(line)
        corpus = " ".join(lines)
        return corpus

    def _read_jp_corpus(self, corpus_path):
        with open(corpus_path, 'r') as f:
            corpus = []
            for line in f:
                line = line.strip().lower()
                if len(line) == 0:
                    continue
                line = self.tagger.parse(line)
                corpus.append(line)
        corpus = " ".join(corpus)
        return corpus

    def sequence2indices(self, sequence):
        indices = []
        for word in sequence.split():
            index = self.word2index[word]
            indices.append(index)
        return indices

    def indices2sequence(self, indices):
        sequence = ''
        for index in indices:
            letter = self.index2word[index]
            sequence += letter
        return sequence