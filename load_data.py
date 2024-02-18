class Data:
    def __init__(
        self,
        data_dir=None,
        train_plus_valid=False,
        reverse=True,
        out_of_vocab_flag=False,
    ):
        """
        ****** reverse=True
        Double the size of datasets by including reciprocal/inverse relations.
        We refer Canonical Tensor Decomposition for Knowledge Base Completion for details

        ****** train_plus_valid=True
        Use the union of training and validation split during training phase.

        ****** out_of_vocab_flag=True
        Remove all triples from validation and test that contain at least one entity that did not occurred during training.

        """
        self.info = {
            "dataset": data_dir,
            "dataset_augmentation": reverse,
            "train_plus_valid": train_plus_valid,
        }

        self.train_data = self.load_data(
            data_dir, split="train", add_reciprical=reverse
        )
        self.valid_data = self.load_data(
            data_dir, split="valid", add_reciprical=reverse
        )
        self.test_data = self.load_data(data_dir, split="test", add_reciprical=False)
        self.data = self.train_data + self.valid_data + self.test_data

        ##Order of entities is important
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)

        self.relations = (
            self.train_relations
            + [i for i in self.valid_relations if i not in self.train_relations]
            + [i for i in self.test_relations if i not in self.train_relations]
        )

        ##Validate for length
        assert set(self.relations) == set(self.train_relations).union(
            set(self.valid_relations).union(set(self.test_relations))
        )

        if train_plus_valid:
            self.train_data.extend(self.valid_data)
            self.valid_data = []

        # Remove triples containing out-of-vocabulary entities from validation and test splits
        if out_of_vocab_flag:
            print(
                "Triples containing out-of-vocabulary entities will be removed from validation and training splits."
            )
            ent = set(self.get_entities(self.train_data))
            print(
                "|G^valid|={0}\t|G^test|={1}".format(
                    len(self.valid_data), len(self.test_data)
                )
            )
            self.valid_data = [
                i for i in self.valid_data if i[0] in ent and i[2] in ent
            ]
            self.test_data = [i for i in self.test_data if i[0] in ent and i[2] in ent]
            print(
                "After removal, |G^valid|={0}\t|G^test|={1}".format(
                    len(self.valid_data), len(self.test_data)
                )
            )

    @staticmethod
    def load_data(data_dir, split, add_reciprical=True):
        print("Loading data from %s/%s.txt" % (data_dir, split))
        try:
            with open("%s/%s.txt" % (data_dir, split), "r") as f:
                data = f.read().strip().split("\n")
                data = [i.split() for i in data]
                if add_reciprical:
                    data += [[i[2], i[1] + "_reverse", i[0]] for i in data]
        except FileNotFoundError as e:
            print(e)
            print("Add empty.")
            raise NotImplementedError("Split not found...")
            # data = []
        return data

    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities
