"""
 Config builder of uLLaVA
 Adapted from: https://github.com/salesforce/LAVIS/blob/main/lavis/common/config.py
"""
import json
from omegaconf import OmegaConf
from utils.registry import registry


class Config:
    def __init__(self, cfg_path):
        self.config = {}

        self.cfg_path = cfg_path

        # Register the config and configuration for setup
        registry.register("configuration", self)

        config = OmegaConf.load(self.cfg_path)

        model_config = self.build_model_config(config)
        dataset_config = self.build_dataset_config(config)
        eval_dataset_config = self.build_eval_dataset_config(config)
        training_config = self.build_training_config(config)
        task_config = self.build_task_config(config)
        processor_config = self.build_processor_config(config)

        # Validate the user-provided runner configuration
        # model and dataset configuration are supposed to be validated by the respective classes
        # [TODO] validate the model/dataset configuration
        # self._validate_runner_config(runner_config)

        # Override the default configuration with user options.
        self.config = OmegaConf.merge(
            model_config, dataset_config, eval_dataset_config, training_config, task_config, processor_config
        )

    @staticmethod
    def build_model_config(config):
        model = config.get("model", None)
        assert model is not None, "Missing model configuration file."

        conv_type = config.get("model", None)
        assert conv_type is not None, "Missing conversation type."

        model_cls = registry.get_model_class(model.arch)
        assert model_cls is not None, f"Model '{model.arch}' has not been registered."

        model_config = OmegaConf.create()
        # hierarchy override, customized config > default config
        model_config = OmegaConf.merge(
            model_config,
            {"model": config["model"]},
        )

        return model_config

    @staticmethod
    def build_dataset_config(config):
        dataset = config.get("dataset", None)
        if dataset is not None:
            dataset_config = OmegaConf.create()

            for dataset_name in dataset:
                # hierarchy override, customized config > default config
                dataset_config = OmegaConf.merge(
                    dataset_config,
                    {"dataset": {dataset_name: config["dataset"][dataset_name]}},
                )
        else:
            empty_config = OmegaConf.create()
            dataset_config = OmegaConf.merge(
                empty_config,
                {"dataset": {}},
            )

        return dataset_config

    @staticmethod
    def build_eval_dataset_config(config):
        eval_dataset = config.get("eval_dataset", None)

        if eval_dataset is not None:
            eval_dataset_config = OmegaConf.create()
            for dataset_name in eval_dataset:
                # hierarchy override, customized config > default config
                eval_dataset_config = OmegaConf.merge(
                    eval_dataset_config,
                    {"eval_dataset": {dataset_name: config["eval_dataset"][dataset_name]}},
                )
        else:
            empty_config = OmegaConf.create()
            eval_dataset_config = OmegaConf.merge(
                empty_config,
                {"eval_dataset": {}},
            )

        return eval_dataset_config

    @staticmethod
    def build_training_config(config):
        training = config.get("training", None)

        if training is None:
            raise KeyError(
                "Expecting 'training' as the root key for dataset configuration."
            )

        training_config = OmegaConf.create()
        # hierarchy override, customized config > default config
        training_config = OmegaConf.merge(
            training_config,
            {"training": config["training"]},
        )

        return training_config

    @staticmethod
    def build_task_config(config):
        task = config.get("task", None)

        if task is None:
            raise KeyError(
                "Expecting 'training' as the root key for dataset configuration."
            )

        task_config = OmegaConf.create()
        # hierarchy override, customized config > default config
        task_config = OmegaConf.merge(
            task_config,
            {"task": config["task"]},
        )

        return task_config

    @staticmethod
    def build_processor_config(config):
        processor = config.get("processor", None)

        if processor is None:
            raise KeyError(
                "Expecting 'training' as the root key for dataset configuration."
            )

        processor_config = OmegaConf.create()
        # hierarchy override, customized config > default config
        processor_config = OmegaConf.merge(
            processor_config,
            {"processor": config["processor"]},
        )

        return processor_config

    def get_config(self):
        return self.config

    def assign_config(self):
        return self.model_cfg, \
               self.dataset_cfg, self.eval_dataset_cfg, self.training_cfg, self.task_cfg, self.processor_cfg

    @property
    def dataset_cfg(self):
        return self.config.dataset

    @property
    def eval_dataset_cfg(self):
        return self.config.eval_dataset

    @property
    def model_cfg(self):
        return self.config.model

    @property
    def training_cfg(self):
        return self.config.training

    @property
    def task_cfg(self):
        return self.config.task

    @property
    def processor_cfg(self):
        return self.config.processor

    def pretty_print(self):
        print(f"\n======  Model Attributes  ======")
        print(self._convert_node_to_json(self.config.model))

        print(f"\n======  Task Attributes  ======")
        print(self._convert_node_to_json(self.config.task))

        print(f"\n======  Processor Attributes  ======")
        print(self._convert_node_to_json(self.config.processor))

        dataset = self.config.dataset
        if dataset is not None:
            print("\n======  Train Dataset Attributes  ======")
            for ds in dataset:
                if ds in self.config.dataset:
                    print(f"\n======== {ds} =======")
                    dataset_config = self.config.dataset[ds]
                    print(self._convert_node_to_json(dataset_config))
                else:
                    print(f"No dataset named '{ds}' in config. Skipping")

        eval_dataset = self.config.get("eval_dataset", None)
        if eval_dataset is not None:
            print("\n======  Evaluate Dataset Attributes  ======")
            for ds in eval_dataset:
                if ds in self.config.eval_dataset:
                    print(f"\n======== {ds} =======")
                    eval_dataset_config = self.config.eval_dataset[ds]
                    print(self._convert_node_to_json(eval_dataset_config))
                else:
                    print(f"No dataset named '{ds}' in config. Skipping")

    @staticmethod
    def _convert_node_to_json(node):
        container = OmegaConf.to_container(node, resolve=True)
        return json.dumps(container, indent=4, sort_keys=True)

    def to_dict(self):
        return OmegaConf.to_container(self.config)


