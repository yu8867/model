import MeCab, unidic_lite
from datasets import load_dataset
from utils.jptoken import JpTextDataset
from torch.utils.data import DataLoader

ds = load_dataset('globis-university/aozorabunko-clean')
tagger = MeCab.Tagger(f"-Owakati -d {unidic_lite.DICDIR}")

# レクチャー動画の解説通り
ds = ds.filter(lambda row: row['meta']['文字遣い種別'] == '新字新仮名')  # 新字新仮名に限定

authors = ['夏目'] #@param
titles =['文鳥', '正岡子規']#@param

corpus = ''
count = 0
for book in ds['train']:
    if count % 500 == 0:
        title = book['meta']['作品名']
        author = book['meta']['姓']

        print(title, author)
    #     # if author in authors:
    #     if title in titles and author in authors:
    #         print(author, title)

        text = book['text'] # 本文
        text = ''.join(text.split()) # Clean up
        text = tagger.parse(text) # 形態素解析
        corpus += text

    count += 1

with open('dataset.txt', 'w') as f:
    f.write(corpus)