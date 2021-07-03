import os
import sys
import json
import collections

import xml.etree.ElementTree as ET
from collections import defaultdict
from os import path
from transformers import BertTokenizerFast
from data_processing.overlap_utils import split_into_segments, get_sentence_map
from data_processing.utils import flatten


class DocumentState(object):
    def __init__(self, key):
        self.doc_key = key
        self.sentence_end = []
        self.token_end = []
        self.tokens = []
        self.subtokens = []
        self.info = []
        self.segments = []
        self.real_segments = []
        self.start_indices = []
        self.end_indices = []
        self.subtoken_map = []
        self.segment_subtoken_map = []
        self.sentence_map = []
        self.pronouns = []
        self.clusters = collections.defaultdict(list)
        self.coref_stacks = collections.defaultdict(list)
        self.speakers = []
        self.segment_info = []

    def finalize(self):
        all_mentions = flatten(self.clusters)
        # print(all_mentions)
        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = flatten(self.segment_subtoken_map)
        # print(len(all_mentions), len(set(all_mentions)))
        assert len(all_mentions) == len(set(all_mentions))
        num_words = len(flatten(self.segments))
        assert num_words == len(subtoken_map), (num_words, len(subtoken_map))
        assert num_words == len(sentence_map), (num_words, len(sentence_map))
        return {
            "doc_key": self.doc_key,
            "sentences": self.segments,
            "real_sentences": self.real_segments,
            "start_indices": self.start_indices,
            "end_indices": self.end_indices,
            "speakers": self.speakers,
            "clusters": self.clusters,
            'sentence_map': sentence_map,
            "subtoken_map": subtoken_map,
        }


def get_document(text_file, xml_file, tokenizer, segment_len):
    document_state = DocumentState(path.basename(text_file))
    doc_word_idx = 0

    sentence_word_map = {}
    for line in open(text_file):
        word = line.strip()
        if word != '':
            doc_word_idx += 1
            sentence_word_map[doc_word_idx] = [len(document_state.subtokens)]

            subtokens = tokenizer.tokenize(word)
            document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]
            for sidx, subtoken in enumerate(subtokens):
                document_state.subtokens.append(subtoken)
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(doc_word_idx)

            sentence_word_map[doc_word_idx].append(len(document_state.subtokens))
            sentence_word_map[doc_word_idx].append(word)
        else:
            document_state.sentence_end[-1] = True

    # Map WikiCoref clusters
    tree = ET.parse(xml_file)
    root = tree.getroot()

    coref_class_to_spans = defaultdict(list)
    # uniq_spans = set()
    for elem in list(root):
        if elem.get('coreftype') == 'ident':
            span = elem.get('span')
            coref_class = elem.get('coref_class')

            # Find span boundaries
            word_span_start, word_span_end = span.split("..")

            word_span_start = int(word_span_start.split("_")[1])
            span_start = sentence_word_map[word_span_start][0]

            word_span_end = int(word_span_end.split("_")[1])
            span_end = sentence_word_map[word_span_end][1] - 1
            coref_class_to_spans[coref_class].append((span_start, span_end))

            # if (span_start, span_end) in uniq_spans:
            #     for idx in range(span_start, span_end + 1):
            #         print(xml_file)
            #         print(word_span_start, word_span_end)
            #         print(elem.get('id'))
            #         print(sentence_word_map[idx][2])
            # else:
            #     uniq_spans.add((span_start, span_end))

    document_state.clusters = list(coref_class_to_spans.values())
    # print(document_state.clusters)

    split_into_segments(document_state, segment_len, document_state.sentence_end, document_state.token_end)
    document = document_state.finalize()
    return document


def minimize_split(seg_len, input_dir, output_dir, split="test"):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    # Create cross validation output dir

    import glob
    text_files = glob.glob(path.join(input_dir, "*/*.txt"))
    xml_files = []
    for text_file in text_files:
        markable_dir = path.join(path.dirname(text_file), "Markables")
        ontonotes_file = glob.glob(path.join(markable_dir, "*_OntoNotes*.xml"))[0]
        xml_files.append(ontonotes_file)

    input_path = path.join(input_dir, "{}.jsonl".format(split))
    output_path = path.join(output_dir, "{}.{}.jsonlines".format(split, seg_len))
    count = 0
    print("Minimizing {}".format(input_path))
    with open(output_path, "w") as output_file:
        for (text_file, xml_file) in zip(text_files, xml_files):
            document = get_document(text_file, xml_file, tokenizer, seg_len)
            output_file.write(json.dumps(document))
            output_file.write("\n")
            count += 1
    print("Wrote {} documents to {}".format(count, output_path))


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for seg_len in [512]:
        minimize_split(seg_len, input_dir, output_dir)
