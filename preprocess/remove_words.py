import sys

sys.path.append('../')


if len(sys.argv) != 2:
	sys.exit("Use: python remove_words.py <dataset>")

datasets = ['MR', 'amazon']
dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")

doc_content_list = []
with open('../data/corpus/' + dataset + '.txt', 'rb') as f:
    for line in f.readlines():
        doc_content_list.append(line.strip().decode('utf-8'))

word_freq = {}  # to remove rare words

for doc_content in doc_content_list:
    temp = doc_content
    words = temp.split()
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

clean_docs = []
for doc_content in doc_content_list:
    temp = doc_content
    words = temp.split()
    doc_words = []
    for word in words:
        if word_freq[word] >= 5:
            doc_words.append(word)
    if len(doc_words) == 0:
        doc_words = temp.split()
    else:
        doc_str = ' '.join(doc_words).strip()
    clean_docs.append(doc_str)

clean_corpus_str = '\n'.join(clean_docs)


with open('../data/corpus/' + dataset + '.clean.txt', 'w', encoding='utf-8') as f:
    f.write(clean_corpus_str)
