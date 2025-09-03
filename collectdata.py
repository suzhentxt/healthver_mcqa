"""Scientific fact-checking dataset. Verifies claims based on citation sentences
using evidence from the cited abstracts. Formatted as a paragraph-level entailment task."""


import datasets
import json


_CITATION = """\
@inproceedings{Sarrouti2021EvidencebasedFO,
    title={Evidence-based Fact-Checking of Health-related Claims},
    author={Mourad Sarrouti and Asma Ben Abacha and Yassine Mrabet and Dina Demner-Fushman},
    booktitle={Conference on Empirical Methods in Natural Language Processing},
    year={2021},
    url={https://api.semanticscholar.org/CorpusID:244119074}
}
"""


_DESCRIPTION = """\
HealthVer is a dataset of public health claims, verified against scientific research articles. For this version of the dataset, we follow the preprocessing from the MultiVerS modeling paper https://github.com/dwadden/multivers, verifying claims against full article abstracts rather than individual sentences. Entailment labels and rationales are included.
"""

_URL = "https://scifact.s3.us-west-2.amazonaws.com/longchecker/latest/data.tar.gz"


def flatten(xss):
    return [x for xs in xss for x in xs]


class HealthVerEntailmentConfig(datasets.BuilderConfig):
    """builderconfig for healthver"""

    def __init__(self, **kwargs):
        """

        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super(HealthVerEntailmentConfig, self).__init__(
            version=datasets.Version("1.0.0", ""), **kwargs
        )


class HealthVerEntailment(datasets.GeneratorBasedBuilder):
    """TODO(healthver): Short description of my dataset."""

    # TODO(healthver): Set up version.
    VERSION = datasets.Version("0.1.0")

    def _info(self):
        # TODO(healthver): Specifies the datasets.DatasetInfo object

        features = {
            "claim_id": datasets.Value("int32"),
            "claim": datasets.Value("string"),
            "abstract_id": datasets.Value("int32"),
            "title": datasets.Value("string"),
            "abstract": datasets.features.Sequence(datasets.Value("string")),
            "verdict": datasets.Value("string"),
            "evidence": datasets.features.Sequence(datasets.Value("int32")),
        }

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                features
                # These are the features of your dataset like images, labels ...
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            citation=_CITATION,
        )

    @staticmethod
    def _read_tar_file(f):
        res = []
        for row in f:
            this_row = json.loads(row.decode("utf-8"))
            res.append(this_row)

        return res

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(healthver): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        archive = dl_manager.download(_URL)
        for path, f in dl_manager.iter_archive(archive):
            # The claims are too similar to paper titles; don't include.
            if path == "data/healthver/corpus.jsonl":
                corpus = self._read_tar_file(f)
                corpus = {x["doc_id"]: x for x in corpus}
            elif path == "data/healthver/claims_train.jsonl":
                claims_train = self._read_tar_file(f)
            elif path == "data/healthver/claims_dev.jsonl":
                claims_validation = self._read_tar_file(f)
            elif path == "data/healthver/claims_test.jsonl":
                claims_test = self._read_tar_file(f)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "claims": claims_train,
                    "corpus": corpus,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "claims": claims_validation,
                    "corpus": corpus,
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "claims": claims_test,
                    "corpus": corpus,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, claims, corpus, split):
        """Yields examples."""
        # Loop over claims and put evidence together with claim.
        id_ = -1  # Will increment to 0 on first iteration.
        for claim in claims:
            evidence = {int(k): v for k, v in claim["evidence"].items()}
            for cited_doc_id in claim["doc_ids"]:
                cited_doc = corpus[cited_doc_id]
                abstract_sents = [sent.strip() for sent in cited_doc["abstract"]]

                if cited_doc_id in evidence:
                    this_evidence = evidence[cited_doc_id]
                    verdict = this_evidence[0][
                        "label"
                    ]  # Can take first evidence since all labels are same.
                    evidence_sents = flatten(
                        [entry["sentences"] for entry in this_evidence]
                    )
                else:
                    verdict = "NEI"
                    evidence_sents = []

                instance = {
                    "claim_id": claim["id"],
                    "claim": claim["claim"],
                    "abstract_id": cited_doc_id,
                    "title": cited_doc["title"],
                    "abstract": abstract_sents,
                    "verdict": verdict,
                    "evidence": evidence_sents,
                }

                id_ += 1
                yield id_, instance