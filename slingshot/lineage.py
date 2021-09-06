class Lineage:
    def __init__(self, clusters):
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __repr__(self):
        return 'Lineage' + str(self.clusters)

    def __iter__(self):
        for c in self.clusters:
            yield c
