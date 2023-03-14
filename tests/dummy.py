from copy import deepcopy
from fastNLP.io import DataBundle

from fastie.dataset import build_dataset

def dummy_ner_dataset() -> DataBundle:
    data_bundle = build_dataset(
        [
            {
                "tokens":
                    ["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "."],
                "entity_mentions":
                    [[[0], "ORG"], [[2], "MISC"], [[6], "MISC"]]
            },
            {
                "tokens":
                    ["Peter", "Blackburn"],
                "entity_mentions":
                    [[[0, 1], "PER"]]
            },
            {
                "tokens":
                    ["BRUSSELS", "1996-08-22"],
                "entity_mentions":
                    [[[0], "LOC"]]
            },
            {
                "tokens":
                    ["The", "European", "Commission", "said", "on", "Thursday", "it", "disagreed", "with", "German",
                     "advice", "to", "consumers", "to", "shun", "British", "lamb", "until", "scientists", "determine",
                     "whether", "mad", "cow", "disease", "can", "be", "transmitted", "to", "sheep", "."],
                "entity_mentions":
                    [[[1, 2], "ORG"], [[9], "MISC"], [[15], "MISC"]]
            },
            {
                "tokens": ["Germany", "\'s", "representative", "to", "the", "European", "Union", "'s", "veterinary",
                           "committee", "Werner", "Zwingmann", "said", "on", "Wednesday", "consumers", "should", "buy",
                           "sheepmeat", "from", "countries", "other", "than", "Britain", "until", "the", "scientific",
                           "advice", "was", "clearer", "."],
                "entity_mentions":
                    [[[0], "LOC"], [[5, 6], "ORG"], [[10, 11], "PER"], [[23], "LOC"]]
            },
            {
                "tokens":
                    ["Rabinovich", "is", "winding", "up", "his", "term", "as", "ambassador", "."],
                "entity_mentions":
                    [[[0], "PER"]]
            },
            {
                "tokens":
                    ["He", "will", "be", "replaced", "by", "Eliahu", "Ben-Elissar", ",", "a", "former", "Israeli",
                     "envoy", "to", "Egypt", "and", "right-wing", "Likud", "party", "politician", "."],
                "entity_mentions":
                    [[[5, 6], "PER"], [[10], "MISC"], [[13], "LOC"], [[16], "ORG"]]
            },
            {
                "tokens":
                    ["Israel", "on", "Wednesday", "sent", "Syria", "a", "message", ",", "via", "Washington", ",",
                     "saying", "it", "was", "committed", "to", "peace", "and", "wanted", "to", "open", "negotiations",
                     "without", "preconditions", "."],
                "entity_mentions":
                    [[[0], "LOC"], [[4], "LOC"], [[9], "LOC"]]
            }
        ]
    )
    data_bundle.set_dataset(deepcopy(data_bundle.get_dataset('train')), "dev")
    data_bundle.set_dataset(deepcopy(data_bundle.get_dataset('train')), "test")
    data_bundle.set_dataset(deepcopy(data_bundle.get_dataset('train')), "infer")
    data_bundle.get_dataset("infer").delete_field("entity_mentions")
    return data_bundle
