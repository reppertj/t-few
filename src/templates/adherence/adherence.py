import json
import os
from typing import List, Tuple

import datasets

TEST_PAPERS = {
    "vittengl-2009.pdf",
    "budtz-jorgensen-2006.pdf",
    "agberotimi-2021.pdf",
    "deutschmann-2019.pdf",
    "chakravorty-bedy-2019.pdf",
    "abebe-2018.pdf",
    "vuuren-cherney-2014.pdf",
    "cheng-2020.pdf",
    "idinsight-2020.pdf",
    "awasthi-2013.pdf",
    "miguel-kremer-2004.pdf",
    "dicko-2008.pdf",
    "routledge-2006.pdf",
    "nyqvist-218.pdf",
    "matangila-2015.pdf",
    "keenan-2018.pdf",
    "bassi-2018.pdf",
    "smithuis-2013.pdf",
}


_DESCRIPTION = """\
Whether a sentence from a research paper is about take-up/adherence/compliance or not."""

_HOMEPAGE = ""

_LICENSE = ""


class AdherenceDataset(datasets.GeneratorBasedBuilder):
    """Adherence Dataset"""

    data_dir = "src/templates/adherence"

    VERSION = datasets.Version("0.1.0")

    def _info(self):
        features = datasets.Features(
            {
                "sentence": datasets.Value("string"),
                "context": datasets.Value("string"),
                "label": datasets.Value("int8"),
                "section": datasets.Value("string"),
                "prediction": datasets.Value("float"),
                "intervention": datasets.Value("string"),
                "document_id": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"split": datasets.Split.TRAIN},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"split": datasets.Split.VALIDATION},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"split": datasets.Split.TEST},
            ),
        ]

    def _generate_examples(self, split):
        source_files = ("train.json", "test.json")
        all_examples = []
        for file in source_files:
            all_examples.extend(list(_generate_from_file(os.path.join(self.data_dir, file), split)))
        
        positive_examples = [example for example in all_examples if example["label"] == 1]
        negative_examples = [example for example in all_examples if example["label"] == 0]

        if datasets.Split.TRAIN == split:
            # get 80% of both positive and negative examples
            positive_train = positive_examples[: int(len(positive_examples) * 0.8)]
            negative_train = negative_examples[: int(len(negative_examples) * 0.8)]
            # shuffle for added security in case the training order isn't random
            train = positive_train + negative_train
            print(f"Training on {len(train)} examples")
            yield from ((idx, example[1]) for idx, example in enumerate(train))

        else:
            # train on the remaining 20% of positive and negative examples
            positive_val = positive_examples[int(len(positive_examples) * 0.8) :]
            negative_val = negative_examples[int(len(negative_examples) * 0.8) :]
            val = positive_val + negative_val
            print(f"Validation on {len(val)} examples")
            yield from ((idx, example[1]) for idx, example in enumerate(val))


def _generate_from_file(filename: str, split):
    with open(filename, "r") as f:
        data = json.load(f)

    rows: List[Tuple[int, dict]] = []

    for row in data:
        label = (
            1
            if row["annotations"]
            and row["annotations"][0].get("result")
            and row["annotations"][0]["result"][0]
            .get("value", {})
            .get("choices", [""])[0] in ("adherence", "1")
            else 0
        )
        rows.append({
            "sentence": row["data"]["sentence"],
            "context": row["data"]["context"],
            "intervention": row["data"]["intervention"],
            "section": row["data"]["section"],
            "document_id": row["data"]["document_id"],
            "prediction": row["data"].get("base_prediction") or row["data"]["prediction"],
            "label": label,
        })

    for row in rows:
        yield row