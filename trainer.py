from tqdm import tqdm
from collections import defaultdict
import json
import numpy as np
import logging
from model import *
from data import *
import os


class Trainer:
    """
    Trainer Class for training and evaluation
    """

    def __init__(self, *, dataset, parameters):

        self.mode = parameters["mode"]
        self.dataset = dataset
        self.embedding_dim = parameters["embedding_dim"]
        self.num_of_epochs = parameters["epochs"]
        self.learning_rate = parameters["learning_rate"]
        self.batch_size = parameters["batch_size"]
        self.label_smoothing = parameters["label_smoothing"]
        self.cuda = torch.cuda.is_available()
        print("Cuda available: ", self.cuda)
        self.optimizer = None
        self.entity_to_idx = None
        self.relation_to_idx = None
        self.negative_label = -1.0
        self.positive_label = 1.0
        self.kwargs = parameters
        self.storage_path = parameters['save_dir'] +"/"+  parameters["dataset"]
        os.makedirs(self.storage_path, exist_ok=True)

        # if "norm_flag" not in self.kwargs:
        #     self.kwargs["norm_flag"] = False

    def get_data_idxs(self, data):
        data_idxs = [
            (
                self.entity_to_idx[data[i][0]],
                self.relation_to_idx[data[i][1]],
                self.entity_to_idx[data[i][2]],
            )
            for i in range(len(data))
        ]
        return data_idxs

    def get_er_vocab(self, data):
        # sub entity and relation
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_re_vocab(self, data):
        # relation and obj entity
        re_vocab = defaultdict(list)
        for triple in data:
            re_vocab[(triple[1], triple[2])].append(triple[0])
        return re_vocab

    def evaluate(self, model):
        """
        Evaluate the trained model on test data
        """
        print("Evaluatin on the test dataset.....")
        model.eval()
        data = self.dataset.test

        model.eval()
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(self.dataset.data))
        re_vocab = self.get_re_vocab(self.get_data_idxs(self.dataset.data))

        for i in tqdm(range(0, len(test_data_idxs)), total=len(test_data_idxs)):
            data_point = test_data_idxs[i]
            e1_idx = torch.tensor(data_point[0])
            rel_idx = torch.tensor(data_point[1])
            e2_idx = torch.tensor(data_point[2])

            if self.cuda:
                e1_idx = e1_idx.cuda()
                rel_idx = rel_idx.cuda()
                e2_idx = e2_idx.cuda()

            all_entities = (
                torch.arange(0, len(self.entity_to_idx)).long().to(e2_idx.device)
            )
            all_entities = all_entities.reshape(
                len(all_entities),
            )

            ################## For OBJECT Prediction ###############
            # (entity, relation) -> use all entities as objects
            obj_predictions = model.predict(
                e1_idx=e1_idx.repeat(
                    len(self.entity_to_idx),
                ),
                rel_idx=rel_idx.repeat(
                    len(self.entity_to_idx),
                ),
                e2_idx=all_entities,
            )

            filt = er_vocab[(data_point[0], data_point[1])]
            target_value = obj_predictions[e2_idx].item()
            obj_predictions[filt] = -np.Inf
            obj_predictions[e1_idx] = -np.Inf
            obj_predictions[e2_idx] = target_value

            sort_values, sort_idxs = torch.sort(obj_predictions, descending=True)
            sort_idxs = sort_idxs.cpu().numpy()
            rank = np.where(sort_idxs == e2_idx.item())[0][0]
            ranks.append(rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

            ################## For SUBJECT Prediction ###############
            # (relation, entity) -> use all entities as subjects
            sub_predictions = model.predict(
                e1_idx=all_entities,
                rel_idx=rel_idx.repeat(
                    len(self.entity_to_idx),
                ),
                e2_idx=e2_idx.repeat(
                    len(self.entity_to_idx),
                ),
                obj=False,
            )

            filt = re_vocab[(data_point[1], data_point[2])]
            target_value = sub_predictions[e1_idx].item()
            sub_predictions[filt] = -np.Inf
            sub_predictions[e2_idx] = -np.Inf
            sub_predictions[e1_idx] = target_value

            # print(obj_predictions.shape)
            # print(sub_predictions.shape)
            # raise NotImplemented

            sort_values, sort_idxs = torch.sort(sub_predictions, descending=True)
            sort_idxs = sort_idxs.cpu().numpy()
            rank = np.where(sort_idxs == e1_idx.item())[0][0]
            ranks.append(rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

        hit_1 = sum(hits[0]) / (float(len(hits[0])))
        hit_3 = sum(hits[2]) / (float(len(hits[0])))
        hit_10 = sum(hits[9]) / (float(len(hits[0])))
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

    def train(self, model):
        """
        Training ComplexE
        """
        if self.cuda:
            model.cuda()
        model.initialize_weights()
        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=self.learning_rate)

        print(f"Starting training.........")
        num_param = sum([p.numel() for p in model.parameters()])
        print(f"Trainable parameters: {num_param}")

        train_data_idxs = self.get_data_idxs(self.dataset.train)
        losses = []

        subject_to_relation_batch = DataLoader(
            BatchLoader(
                er_vocab=self.get_er_vocab(train_data_idxs),
                num_e=len(self.dataset.entities),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        loss_of_epoch, it = -1, -1
        looper = tqdm(
            range(1, self.num_of_epochs + 1),
            desc=f"Training: ",
            total=self.num_of_epochs,
        )
        for it in looper:
            loss_of_epoch = 0.0
            for batch in subject_to_relation_batch:
                e1_idx, r_idx, targets = batch
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                    targets = targets.cuda()

                # Regularization
                # https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06
                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (
                        self.label_smoothing / targets.size(1)
                    )

                self.optimizer.zero_grad()
                pred, loss = model(e1_idx, r_idx, targets)
                loss_of_epoch += loss.item()
                loss.backward()
                self.optimizer.step()

            losses.append(loss_of_epoch)
            looper.set_postfix_str(f"Loss at {it} epoch: {loss_of_epoch:.4f}")

        # Save model
        torch.save(model.state_dict(), self.storage_path + "/model.pt")

    def main(self):
        """
        Train and evaluate the model
        """

        self.entity_to_idx = {
            self.dataset.entities[i]: i for i in range(len(self.dataset.entities))
        }
        self.relation_to_idx = {
            self.dataset.relations[i]: i for i in range(len(self.dataset.relations))
        }

        self.kwargs.update(
            {
                "num_entities": len(self.entity_to_idx),
                "num_relations": len(self.relation_to_idx),
            }
        )

        # Intialize the model
        model = ComplexEx(self.kwargs)
        if self.mode == "train":
            self.train(model)
        else:
            # Load model
            print("Loading the model.....")
            checkpoint = torch.load(self.storage_path + "/model.pt")
            model.load_state_dict(checkpoint)
            if self.cuda:
                model.cuda()

        results = self.evaluate(model)
        with open(f"{self.storage_path}/results.json", "w") as f:
            num_param = sum([p.numel() for p in model.parameters()])
            results["Number_param"] = num_param
            results.update(self.kwargs)
            json.dump(results, f, indent=4)
