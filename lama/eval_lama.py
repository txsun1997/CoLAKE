import argparse
import pprint
import statistics
from os import listdir
import os
from os.path import isfile, join
from shutil import copyfile
from collections import defaultdict
import sys
sys.path.append('../')
from lama.lama_utils import load_file
from lama.model import Roberta
from lama.batch_eval_KB_completion import run_evaluation

common_vocab_path = "../data/LAMA/common_vocab_cased.txt"
model_path = "../model/"

def get_TREx_parameters(data_path_pre="../data/LAMA/"):
    relations = load_file("{}relations.jsonl".format(data_path_pre))
    data_path_pre += "TREx/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_GoogleRE_parameters():
    relations = [
        {
            "relation": "place_of_birth",
            "template": "[X] was born in [Y] .",
            "template_negated": "[X] was not born in [Y] .",
        },
        {
            "relation": "date_of_birth",
            "template": "[X] (born [Y]).",
            "template_negated": "[X] (not born [Y]).",
        },
        {
            "relation": "place_of_death",
            "template": "[X] died in [Y] .",
            "template_negated": "[X] did not die in [Y] .",
        },
    ]
    data_path_pre = "../data/LAMA/Google_RE/"
    data_path_post = "_test.jsonl"
    return relations, data_path_pre, data_path_post


def eval_model(relations, data_path_pre, data_path_post):
    all_Precision1 = []
    type_Precision1 = defaultdict(list)
    type_count = defaultdict(list)

    for relation in relations:
        PARAMETERS = {
            "dataset_filename": "{}{}{}".format(
                data_path_pre, relation["relation"], data_path_post
            ),
            "common_vocab_filename": common_vocab_path,
            "template": "",
            "batch_size": 64,
            "max_sentence_length": 100,
            "threads": -1,
            "model_path": model_path
        }

        if "template" in relation:
            PARAMETERS["template"] = relation["template"]

        print(PARAMETERS)

        args = argparse.Namespace(**PARAMETERS)

        # see if file exists
        try:
            data = load_file(args.dataset_filename)
        except Exception as e:
            print("Relation {} excluded.".format(relation["relation"]))
            print("Exception: {}".format(e))
            continue

        model = Roberta(args)
        print("Model: {}".format(model.__class__.__name__))

        Precision1 = run_evaluation(args, shuffle_data=False, model=model)
        print("P@1 : {}".format(Precision1), flush=True)
        all_Precision1.append(Precision1)

        if "type" in relation:
            type_Precision1[relation["type"]].append(Precision1)
            data = load_file(PARAMETERS["dataset_filename"])
            type_count[relation["type"]].append(len(data))

    mean_p1 = statistics.mean(all_Precision1)
    print("@@@ mean P@1: {}".format(mean_p1))

    for t, l in type_Precision1.items():
        print(
            "@@@ ",
            t,
            statistics.mean(l),
            sum(type_count[t]),
            len(type_count[t]),
            flush=True,
        )

    return mean_p1, all_Precision1


if __name__ == "__main__":
    print("1. Google-RE")
    parameters = get_GoogleRE_parameters()
    eval_model(*parameters)

    print("2. T-REx")
    parameters = get_TREx_parameters()
    eval_model(*parameters)
