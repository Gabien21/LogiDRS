# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""


import logging
import os
import json
from dataclasses import dataclass, field
from typing import Dict, Optional
from plot_result import plot_image

import numpy as np

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer, AdamW,
    TrainingArguments,
    set_seed,
)
from transformers import TrainerCallback, TrainerControl, TrainerState
from torch.utils.data import Subset

from utils_multiple_choice import processors
from collections import Counter

from dagn2 import DAGN
from tokenization_dagn import arg_tokenizer
# from utils_multiple_choice import Split, MyMultipleChoiceDataset
from graph_building_blocks.argument_set_punctuation_v4 import punctuations
with open('./graph_building_blocks/explicit_arg_set_v4.json', 'r') as f:
    relations = json.load(f)  # key: relations, value: ignore



logger = logging.getLogger(__name__)


class TestEvaluationCallback(TrainerCallback):
    def __init__(self, trainer, test_dataset, output_dir, task_name):
        self.trainer = trainer
        self.test_dataset = test_dataset
        self.output_dir = output_dir
        self.task_name = task_name

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Gọi sau mỗi epoch. Nếu epoch >= 10, thực hiện dự đoán trên test set."""
        # if state.epoch >= 10:
        if True:
            print(f"\n[Epoch {state.epoch}] Running test evaluation for {self.task_name}...\n")
            
            test_result = self.trainer.predict(self.test_dataset)
            preds = test_result.predictions
            pred_ids = np.argmax(preds, axis=1)

            output_epoch_path = os.path.join(self.output_dir, f"test_predictions_epoch_{int(state.epoch)}")

            if self.task_name == "reclor":
                # Lưu predictions dưới dạng .npy
                np.save(f"{output_epoch_path}.npy", pred_ids)
                print(f"ReClor predictions for epoch {int(state.epoch)} saved to {output_epoch_path}.npy")

            elif self.task_name in ["logiqa", "logiqa2"]:
                # Lưu kết quả test dưới dạng file text
                output_test_file = f"{output_epoch_path}_results.txt"
                with open(output_test_file, "w") as writer:
                    for key, value in test_result.metrics.items():
                        writer.write(f"{key} = {value}\n")
                print(f"{self.task_name} results for epoch {int(state.epoch)} saved to {output_test_file}")

            else:
                print(f"⚠ Warning: Task {self.task_name} is not explicitly handled. Saving generic results.")
                np.save(f"{output_epoch_path}.npy", pred_ids)



def simple_accuracy(preds, labels):
    return (preds == labels).mean()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_type: str = field(
        metadata={"help": "Model types: roberta_large | argument_numnet | ..."},
        default="DAGN"
    )
    merge_type: int = field(
        default=1,
        metadata={"help": "The way gcn_feats and baseline_feats are merged."}
    )
    gnn_version: str = field(
        default="",
        metadata={"help": "GNN version in myutil.py or myutil_gat.py"
                          "value = GCN|GCN_reversededges|GCN_reversededges_double"}
    )
    model_branch: bool = field(
        default=False,
        metadata={"help": "add model branch according to grouped_question_type"}
    )
    model_version: int = field(
        default=1,
        metadata={"help": "argument numnet evolving version."}
    )
    use_gcn: bool = field(
        default=False,
        metadata={"help": "Use GCN in model or not."}
    )
    use_discourse: bool = field(
        default=False,
        metadata={"help": "Use discourse information in model or not."}
    )
    use_transformer: bool = field(
        default=False,
        metadata={"help": "Use transformer information in model or not."}
    )
    use_pool: bool = field(
        default=False,
        metadata={"help": "Use pooled_output branch in model or not."}
    )
    gcn_steps: int = field(
        default=1,
        metadata={"help": "GCN iteration steps"}
    )
    attention_drop: float = field(
        default=0.1,
        metadata={"help": "huggingface RoBERTa config.attention_probs_dropout_prob"}
    )
    hidden_drop: float = field(
        default=0.1,
        metadata={"help": "huggingface RoBERTa config.hidden_dropout_prob"}
    )
    numnet_drop: float = field(
        default=0.1,
        metadata={"help": "NumNet dropout probability"}
    )
    init_weights: bool = field(
        default=False,
        metadata={"help": "init weights in Argument NumNet."}
    )

    # training
    roberta_lr: float = field(
        default=5e-6,
        metadata={"help": "learning rate for updating roberta parameters"}
    )
    gcn_lr: float = field(
        default=5e-6,
        metadata={"help": "learning rate for updating gcn parameters"}
    )
    proj_lr: float = field(
        default=5e-6,
        metadata={"help": "learning rate for updating fc parameters"}
    )



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())})
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    data_type: str = field(
        default="argument_numnet",
        metadata={
            "help": "data types in utils script. roberta_large | argument_numnet "
        }
    )
    graph_building_block_version: int = field(
        default=2,
        metadata={
            "help": "graph building block version."
        }
    )
    data_processing_version: int = field(
        default=2,
        metadata={
            "help": "data processing version"
        }
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        }
    )
    max_ngram: int = field(
        default=5,
        metadata={"help": "max ngram when pre-processing text and find argument/domain words."}
        )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    demo_data: bool = field(
        default=False,
        metadata={"help": "demo data sets with 100 samples."}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        processor = processors[data_args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))


    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )




    if isinstance(relations, tuple): max_rel_id = int(max(relations[0].values()))
    elif isinstance(relations, dict):
        if not len(relations) == 0:
            max_rel_id = int(max(relations.values()))
        else:
            max_rel_id = 0
    else: raise Exception

    if model_args.model_type == "PLM":
        from utils_multiple_choice_plm import Split, MultipleChoiceDataset
        model = AutoModelForMultipleChoice.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        train_dataset = (
            MultipleChoiceDataset(
                data_dir=data_args.data_dir,
                tokenizer=tokenizer,
                task=data_args.task_name,
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
                mode=Split.train,
                demo=data_args.demo_data
            )
            if training_args.do_train
            else None
        )
        eval_dataset = (
            MultipleChoiceDataset(
                data_dir=data_args.data_dir,
                tokenizer=tokenizer,
                task=data_args.task_name,
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
                mode=Split.dev,
                demo=data_args.demo_data
            )
            if training_args.do_eval
            else None
        )
        test_dataset = (
            MultipleChoiceDataset(
                data_dir=data_args.data_dir,
                tokenizer=tokenizer,
                task=data_args.task_name,
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
                mode=Split.test,
                demo=data_args.demo_data
            )
            if training_args.do_predict
            else None
        )
    elif model_args.model_type == "DAGN":
        from utils_multiple_choice import Split, MyMultipleChoiceDataset
        training_args.report_to = []
       
        model = DAGN.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            token_encoder_type="roberta" if "roberta" in model_args.model_name_or_path else "bert",
            init_weights=model_args.init_weights,
            max_rel_id=max_rel_id,
            merge_type=model_args.merge_type,
            gnn_version=model_args.gnn_version,
            cache_dir=model_args.cache_dir,
            hidden_size=config.hidden_size,
            dropout_prob=model_args.numnet_drop,
            use_discourse=model_args.use_discourse,
            use_transformer=model_args.use_transformer,
            use_gcn=model_args.use_gcn,
            use_pool=model_args.use_pool,
            gcn_steps=model_args.gcn_steps
        )

        train_dataset = (
            MyMultipleChoiceDataset(
                data_dir=data_args.data_dir,
                tokenizer=tokenizer,
                arg_tokenizer=arg_tokenizer,
                data_processing_version=data_args.data_processing_version,
                graph_building_block_version=data_args.graph_building_block_version,
                relations=relations,
                punctuations=punctuations,
                task=data_args.task_name,
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
                mode=Split.train,
                demo=data_args.demo_data
            )
            if training_args.do_train
            else None
        )
        eval_dataset = (
            MyMultipleChoiceDataset(
                data_dir=data_args.data_dir,
                tokenizer=tokenizer,
                arg_tokenizer=arg_tokenizer,
                data_processing_version=data_args.data_processing_version,
                graph_building_block_version=data_args.graph_building_block_version,
                relations=relations,
                punctuations=punctuations,
                task=data_args.task_name,
                max_seq_length=data_args.max_seq_length,
                max_ngram=data_args.max_ngram,
                overwrite_cache=data_args.overwrite_cache,
                mode=Split.dev,
                demo=data_args.demo_data
            )
            if training_args.do_eval
            else None
        )
        test_dataset = (
            MyMultipleChoiceDataset(
                data_dir=data_args.data_dir,
                tokenizer=tokenizer,
                arg_tokenizer=arg_tokenizer,
                data_processing_version=data_args.data_processing_version,
                graph_building_block_version=data_args.graph_building_block_version,
                relations=relations,
                punctuations=punctuations,
                task=data_args.task_name,
                max_seq_length=data_args.max_seq_length,
                max_ngram=data_args.max_ngram,
                overwrite_cache=data_args.overwrite_cache,
                mode=Split.test,
                demo=data_args.demo_data
            )
            if training_args.do_predict
            else None
        )
        train_dataset = Subset(train_dataset, range(10))
        eval_dataset = Subset(eval_dataset, range(10))
        test_dataset = Subset(test_dataset, range(10))

    else:
        raise Exception


    training_args.report_to = []

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": simple_accuracy(preds, p.label_ids)}


    if model_args.use_gcn:
        no_decay = ["bias", "LayerNorm.weight"]
        # model = model_init()
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if n.startswith("_gcn")
                           and not any(nd in n for nd in no_decay)],
                "lr": model_args.gcn_lr,
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if n.startswith("_gcn")
                           and any(nd in n for nd in no_decay)],
                "lr": model_args.gcn_lr,
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if n.startswith("roberta")
                           and not any(nd in n for nd in no_decay)],
                "lr": model_args.roberta_lr,
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if n.startswith("roberta")
                           and any(nd in n for nd in no_decay)],
                "lr": model_args.roberta_lr,
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if n.startswith("_proj")
                           and not any(nd in n for nd in no_decay)],
                "lr": model_args.proj_lr,
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if n.startswith("_proj")
                           and any(nd in n for nd in no_decay)],
                "lr": model_args.proj_lr,
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if n.startswith("_discourse")
                           and not any(nd in n for nd in no_decay)],
                "lr": model_args.proj_lr,
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if n.startswith("_discourse")
                           and any(nd in n for nd in no_decay)],
                "lr": model_args.proj_lr,
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if n.startswith("_transformer")
                           and not any(nd in n for nd in no_decay)],
                "lr": model_args.proj_lr,
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if n.startswith("_transformer")
                           and any(nd in n for nd in no_decay)],
                "lr": model_args.proj_lr,
                "weight_decay": 0.0,
            }
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
        )
        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, None),
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
    
    trainer.add_callback(TestEvaluationCallback(trainer, test_dataset, training_args.output_dir, data_args.task_name))
    
    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        # trainer.save_model()
        trainer_state_path = os.path.join(training_args.output_dir, "trainer_state.json")
        trainer.state.save_to_json(trainer_state_path)
        print(f"Trainer state saved to {trainer_state_path}")
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

                results.update(result)

    def test_all_checkpoints(output_dir, trainer, test_dataset, model_type="DAGN"):
        """
        Function to evaluate all checkpoints saved in the output directory.
        """
        # Lấy danh sách tất cả các checkpoints trong output_dir
        checkpoints = sorted(
            [os.path.join(output_dir, ckpt) for ckpt in os.listdir(output_dir) if ckpt.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1])  # Sắp xếp theo số bước (epoch/step)
        )

        results = {}

        # Lặp qua từng checkpoint và đánh giá
        for checkpoint in checkpoints:
            checkpoint_name = os.path.basename(checkpoint)  # Lấy tên checkpoint
            print(f"Loading model from {checkpoint}")

            # Load model tương ứng
            if model_type == "PLM":
                model = AutoModelForMultipleChoice.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                )
            else:  # Model DAGN
                model = DAGN.from_pretrained(
                    checkpoint,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    token_encoder_type="roberta" if "roberta" in model_args.model_name_or_path else "bert",
                    init_weights=model_args.init_weights,
                    max_rel_id=max_rel_id,
                    merge_type=model_args.merge_type,
                    gnn_version=model_args.gnn_version,
                    cache_dir=model_args.cache_dir,
                    hidden_size=config.hidden_size,
                    dropout_prob=model_args.numnet_drop,
                    use_discourse=model_args.use_discourse,
                    use_transformer=model_args.use_transformer,
                    use_gcn=model_args.use_gcn,
                    use_pool=model_args.use_pool,
                    gcn_steps=model_args.gcn_steps
                )
            print("this is model------------------------------------")
            print(model)
            print("this is model------------------------------------")
            trainer.model = model  # Cập nhật model cho trainer
            model.to(training_args.device)

            print(f"*** Evaluating {checkpoint} ***")
            # eval_result = trainer.predict(eval_dataset)  # Đánh giá
            # print(eval_result.metrics)

            # Trường hợp đặc biệt cho task "reclor"
            if data_args.task_name == "reclor":
                test_result = trainer.predict(test_dataset)
                preds = test_result.predictions  
                pred_ids = np.argmax(preds, axis=1)

                # Lưu predictions với task_name và checkpoint
                output_test_file = os.path.join(
                    checkpoint, f"{os.path.basename(os.path.dirname(training_args.output_dir))}_{checkpoint_name}_predictions.npy"
                )
                np.save(output_test_file, pred_ids)
                print(f"Predictions for {checkpoint} saved to {output_test_file}")

            else:
                # Trường hợp thông thường
                test_result = trainer.predict(test_dataset)

                # Lưu kết quả đánh giá với task_name và checkpoint
                output_test_file = os.path.join(
                    checkpoint, f"{os.path.basename(os.path.dirname(training_args.output_dir))}_{checkpoint_name}_test_results.txt"
                )
                with open(output_test_file, "w") as writer:
                    for key, value in test_result.metrics.items():
                        writer.write(f"{key} = {value}\n")
                print(f"Results for {checkpoint} saved to {output_test_file}")

                # Lưu kết quả vào dict
                results[f"{os.path.basename(os.path.dirname(training_args.output_dir))}_{checkpoint_name}"] = test_result.metrics

        return results



    # Test
    if training_args.do_predict:
        if data_args.task_name == "reclor":
            logger.info("*** Test ***")

            test_result = trainer.predict(test_dataset)
            preds = test_result.predictions  # np array. (1000, 4)
            pred_ids = np.argmax(preds, axis=1)

            output_test_file = os.path.join(training_args.output_dir,  f"{os.path.basename(os.path.dirname(training_args.output_dir))}_predictions.npy")
            np.save(output_test_file, pred_ids)
            logger.info("predictions saved to {}".format(output_test_file))
        else :
            logger.info("*** Test ***")

            test_result = trainer.predict(test_dataset)

            output_test_file = os.path.join(training_args.output_dir, f"{os.path.basename(os.path.dirname(training_args.output_dir))}test_results.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results *****")
                    for key, value in test_result.metrics.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

                    results.update(test_result.metrics)

            
        print("Testing all checkpoints...")
        all_results = test_all_checkpoints(training_args.output_dir, trainer, test_dataset, model_args.model_type)
        if data_args.task_name != "reclor":
            test_acc_results = {checkpoint: metrics["test_acc"] for checkpoint, metrics in all_results.items()}
            output_acc_file = os.path.join(training_args.output_dir, "all_checkpoints_test_acc.json")
            with open(output_acc_file, "w") as f:
                json.dump(test_acc_results, f, indent=4)
        print("All checkpoint results:", all_results)
            
    try:
        output_image_file = os.path.join(training_args.output_dir, os.path.basename(os.path.dirname(training_args.output_dir))+".png")
        plot_image(training_args.output_dir, output_image_file)
        print(f"Result saved to {output_image_file}")
    except Exception as e:
        print(e)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()
    


if __name__ == "__main__":
    main()
