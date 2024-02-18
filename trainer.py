from tqdm import tqdm
from collections import defaultdict
import json
import numpy as np
import logging
from model import *
from data import *


class Trainer:
    """
    Trainer Class for training and evaluation
    """

    def __init__(self, *, dataset, model, parameters):

        self.dataset = dataset
        self.model = model
        self.embedding_dim = parameters["embedding_dim"]
        self.num_of_epochs = parameters["epochs"]
        self.learning_rate = parameters["learning_rate"]
        self.batch_size = parameters["batch_size"]
        self.label_smoothing = parameters["label_smoothing"]
        self.cuda = torch.cuda.is_available()
        print("Cuda available: ", self.cuda)
        self.num_of_workers = parameters["num_workers"]
        self.optimizer = None
        self.entity_idxs = None
        self.relation_idxs = None
        self.negative_label = 0.0
        self.positive_label = 1.0
        self.kwargs = parameters
        self.kwargs["model"] = self.model
        self.storage_path = parameters["dataset"]

        # if "norm_flag" not in self.kwargs:
        #     self.kwargs["norm_flag"] = False

    def get_data_idxs(self, data):
        data_idxs = [
            (
                self.entity_idxs[data[i][0]],
                self.relation_idxs[data[i][1]],
                self.entity_idxs[data[i][2]],
            )
            for i in range(len(data))
        ]
        return data_idxs

    @staticmethod
    def get_er_vocab(data):
        # sub entity and relation
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    @staticmethod
    def get_re_vocab(data):
        # relation and obj entity
        re_vocab = defaultdict(list)
        for triple in data:
            re_vocab[(triple[1], triple[2])].append(triple[0])
        return re_vocab

    def get_batch_1_to_N(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx : idx + self.batch_size]
        targets = (
            np.ones((len(batch), len(self.dataset.entities))) * self.negative_label
        )
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = self.positive_label
        return np.array(batch), torch.FloatTensor(targets)

    def evaluate_one_to_n(self, model, data, log_info="Evaluate one to N."):
        """
        Evaluate model
        """
        print(log_info)
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])
        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(self.dataset.data))

        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch_1_to_N(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:, 0])
            r_idx = torch.tensor(data_batch[:, 1])
            e2_idx = torch.tensor(data_batch[:, 2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.forward_head_batch(e1_idx=e1_idx, rel_idx=r_idx)
            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j, e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                ranks.append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)

        hit_1 = sum(hits[0]) / (float(len(data)))
        hit_3 = sum(hits[2]) / (float(len(data)))
        hit_10 = sum(hits[9]) / (float(len(data)))
        mean_rank = np.mean(ranks)
        mean_reciprocal_rank = np.mean(1.0 / np.array(ranks))

        print(f"Hits @10: {hit_10}")
        print(f"Hits @3: {hit_3}")
        print(f"Hits @1: {hit_1}")
        print(f"Mean rank: {mean_rank}")
        print(f"Mean reciprocal rank: {mean_reciprocal_rank}")

        results = {
            "H@1": hit_1,
            "H@3": hit_3,
            "H@10": hit_10,
            "MR": mean_rank,
            "MRR": mean_reciprocal_rank,
        }

        return results

    def evaluate_standard(self, model, data, log_info="Evaluate one to N."):
        print(log_info)
        # model.to('cpu')
        model.eval()
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(self.dataset.data))

        for i in range(0, len(test_data_idxs)):
            data_point = test_data_idxs[i]
            e1_idx = torch.tensor(data_point[0])
            rel_idx = torch.tensor(data_point[1])
            e2_idx = torch.tensor(data_point[2])

            if self.cuda:
                e1_idx = e1_idx.cuda()
                rel_idx = rel_idx.cuda()
                e2_idx = e2_idx.cuda()

            all_entities = (
                torch.arange(0, len(self.entity_idxs)).long().to(e2_idx.device)
            )
            all_entities = all_entities.reshape(
                len(all_entities),
            )

            # For single entity, relation -> use all entities as targets
            predictions = model.forward_triples(
                e1_idx=e1_idx.repeat(
                    len(self.entity_idxs),
                ),
                rel_idx=rel_idx.repeat(
                    len(self.entity_idxs),
                ),
                e2_idx=all_entities,
            )

            filt = er_vocab[(data_point[0], data_point[1])]
            target_value = predictions[e2_idx].item()
            predictions[filt] = -np.Inf
            predictions[e1_idx] = -np.Inf
            predictions[e2_idx] = target_value

            sort_values, sort_idxs = torch.sort(predictions, descending=True)
            sort_idxs = sort_idxs.cpu().numpy()
            rank = np.where(sort_idxs == e2_idx.item())[0][0]
            ranks.append(rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

        hit_1 = sum(hits[0]) / (float(len(data)))
        hit_3 = sum(hits[2]) / (float(len(data)))
        hit_10 = sum(hits[9]) / (float(len(data)))
        mean_rank = np.mean(ranks)
        mean_reciprocal_rank = np.mean(1.0 / np.array(ranks))
        # with ope
        print(f"Hits @10: {hit_10}")
        print(f"Hits @3: {hit_3}")
        print(f"Hits @1: {hit_1}")
        print(f"Mean rank: {mean_rank}")
        print(f"Mean reciprocal rank: {mean_reciprocal_rank}")

        results = {
            "H@1": hit_1,
            "H@3": hit_3,
            "H@10": hit_10,
            "MR": mean_rank,
            "MRR": mean_reciprocal_rank,
        }

        return results

    def eval(self, model):
        """
        Evaluate the trained model on test data
        """
        if self.dataset.test_data:
            # if self.kwargs['scoring_technique'] == 'KvsAll':
            #     results = self.evaluate_one_to_n(
            #         model,
            #         self.dataset.test_data,
            #         'Standard Link Prediction evaluation on Testing Data')
            # elif self.neg_sample_ratio > 0:

            results = self.evaluate_standard(
                model,
                self.dataset.test_data,
                "Standard Link Prediction evaluation on Testing Data",
            )
            # else:
            # raise ValueError

            with open("results.json", "w") as file_descriptor:
                num_param = sum([p.numel() for p in model.parameters()])
                results["Number_param"] = num_param
                results.update(self.kwargs)
                json.dump(results, file_descriptor, indent=4)

    def val(self, model):
        """
        Validation
        """
        model.eval()
        if self.dataset.valid_data:
            self.evaluate_one_to_n(
                model,
                self.dataset.valid_data,
                "KvsAll Link Prediction validation on Validation",
            )
        model.train()

    def train(self, model):
        """
        Training ComplexE
        """
        if self.cuda:
            model.cuda()
        model.initialize_weights()
        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=self.learning_rate)

        print(f"Starting {model.name} training.........")
        num_param = sum([p.numel() for p in model.parameters()])
        print(f"Trainable parameters: {num_param}")

        # Dump Experiment param
        with open("params.json", "w") as file_descriptor:
            json.dump(self.kwargs, file_descriptor, indent=4)

        model = self.k_vs_all_training_schema(model)

        # Save model
        torch.save(model.state_dict(), self.storage_path + "/model.pt")

    def train_and_eval(self):
        """
        Train and evaluate the model
        """

        self.entity_idxs = {
            self.dataset.entities[i]: i for i in range(len(self.dataset.entities))
        }
        self.relation_idxs = {
            self.dataset.relations[i]: i for i in range(len(self.dataset.relations))
        }

        self.kwargs.update(
            {
                "num_entities": len(self.entity_idxs),
                "num_relations": len(self.relation_idxs),
            }
        )
        self.kwargs.update(self.dataset.info)
        model = ComplexEx(self.kwargs)
        self.train(model)
        self.eval(model)

    def k_vs_all_training_schema(self, model):
        print("k_vs_all_training_schema starts")

        train_data_idxs = self.get_data_idxs(self.dataset.train_data)
        losses = []

        subject_to_relation_batch = DataLoader(
            SubjectAndRelationBatchLoader(
                er_vocab=self.get_er_vocab(train_data_idxs),
                num_e=len(self.dataset.entities),
            ),
            batch_size=self.batch_size,
            num_workers=self.num_of_workers,
            shuffle=True,
        )

        loss_of_epoch, it = -1, -1
        for it in tqdm(
            range(1, self.num_of_epochs + 1),
            desc=f"Training: ",
            total=self.num_of_epochs,
        ):
            loss_of_epoch = 0.0
            # given a triple (e_i,r_k,e_j)
            # we generate two sets of corrupted triples
            # 1) (e_i,r_k,x) where x \in Entities
            # 2) (e_i,r_k,x) \not \in KG
            for sub_batch in subject_to_relation_batch:
                e1_idx, r_idx, targets = sub_batch
                if self.cuda:
                    targets = targets.cuda()
                    r_idx = r_idx.cuda()
                    e1_idx = e1_idx.cuda()

                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (
                        self.label_smoothing / targets.size(1)
                    )

                self.optimizer.zero_grad()
                loss = model.forward_subject_and_loss(e1_idx, r_idx, targets)
                loss_of_epoch += loss.item()
                loss.backward()
                self.optimizer.step()

            losses.append(loss_of_epoch)
            print("Loss at {0}.th epoch:{1}".format(it, loss_of_epoch))
        np.savetxt(
            fname=self.storage_path + "/loss_per_epoch.csv",
            X=np.array(losses),
            delimiter=",",
        )
        model.eval()
        return model
