class Data:
    def __init__(
        self,
        path=None,
        out_of_vocab=False,
    ):
        self.train = self.load_data(path, split="train")
        self.validation = self.load_data(path, split="valid")
        self.test = self.load_data(path, split="test")
        self.data = self.train + self.validation + self.test

        ##Order of entities is important
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train)
        self.valid_relations = self.get_relations(self.validation)
        self.test_relations = self.get_relations(self.test)

        self.relations = (
            self.train_relations
            + [i for i in self.valid_relations if i not in self.train_relations]
            + [i for i in self.test_relations if i not in self.train_relations]
        )

        ##Validate for length
        assert set(self.relations) == set(self.train_relations).union(
            set(self.valid_relations).union(set(self.test_relations))
        )

        # Remove triples containing out-of-vocabulary entities from validation and test splits
        if out_of_vocab:
            ent = set(self.get_entities(self.train))
            self.validation = [
                i for i in self.validation if i[0] in ent and i[2] in ent
            ]
            self.test = [i for i in self.test if i[0] in ent and i[2] in ent]

    def load_data(self, path, split):
        print("Loading data from %s/%s.txt" % (path, split))
        data = []
        try:
            with open("%s/%s.txt" % (path, split), "r") as f:
                data = f.read().strip().split("\n")
                data = [i.split() for i in data]
        except Exception as e:
            print(e)
            raise FileNotFoundError("Split not found...")
        return data

    def get_entities(self, data):
        sub = list(triple[0] for triple in data)
        obj = list(triple[2] for triple in data)
        entities = sorted(list(set(sub + obj)))
        return entities

    def get_relations(self, data):
        rel = list(triple[1] for triple in data)
        relations = sorted(list(set(rel)))
        return relations
