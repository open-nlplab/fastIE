_help = "使用 bert 对 conll2003 数据集进行序列标注"
config = dict(
    task="ner/bert",
    dataset="conll2003",
    num_labels=4,
    tag_vocab = {0: "A", 1: "B", 2:"C", 3:"D"}
)
