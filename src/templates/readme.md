# Adding a Dataset

## Define the Dataset

1. Create a subdirectory here named after the dataset.
2. In that directory, create a `${DATASET_NAME}.py` (matching the name of the directory) file. It should contain a class subclassed from `datasets.GeneratorBasedBuilder`. Model it after `adherence.adherence.AdherenceDataset`. It needs to contain:

```py3
class MyDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        ...
        return datasets.DatasetInfo(...)

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.DownloadManager]:
        # HF Datasets allows defining arbitrary splits
        # However, this repo assumes that we will use the splits as defined in
        # adherence.adherence.AdherenceDataset
        ...

    def _generate_examples(self, split: datasets.Split) -> Generator[Tuple[int, Dict[str, Any]], None, None]
        # This is where you create your train / validation / test split
        # this is also where we read in data from source files, the internet, wherever
        # They should return tuples of (idx, EXAMPLE)
        # where idx monotonically increases from 0 for each split
        # where EXAMPLE matches the schema returned in DatasetInfo in `info`j
        ...

```

## Define templates

1. Create a `templates.yaml` modeled after `src/templates/adherence/templates.yaml`. The important parts are `answer_choices` (` ||| ` separated strings matching the order of integer labels in your dataset). For examples, if your choices are `no ||| yes`, then `0` means "no" and `1` means "yes".
2. Each template should have a UUID.
3. `jinja` should be a string with `{{ field_name }}` placeholders to be filled in from the dataset fields.
4. The target completion is defined after a `|||` in the jinja template, with `{{ answer_choices[label] }}`
5. The metadata field needs to be filled out for compatibility but is ignored.
6. You can include multiple templates.

## Create a configuration file

1. Create a JSON file modeled after `configs/ia3.json`. Set `dataset: "custom"` and `custom_dataset_dir: "src/templates/${DATASET_NAME}"`.
2. Follow the instructions in the root README to train the model using this configuration.
