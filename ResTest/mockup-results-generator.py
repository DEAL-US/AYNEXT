from random import random

class Technique():

    def get_name(self):
        return self.name

    def score(self,source, target, relation) -> float:
        raise NotImplementedError("This function is not implemented in the base class. Please, use other classes that extend it")

class AllYes(Technique):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'AllYes'

    def score(self,source, target, relation) -> float:
        return 1.0

class AllNo(Technique):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'AllNo'

    def score(self,source, target, relation) -> float:
        return 0.0

class AllRandom(Technique):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'AllRandom'

    def score(self,source, target, relation) -> float:
        return round(random(), 2)

dataset_file = '../DataGen/WN11-dataset/test.txt'
output_file = './mockup-results.txt'

techniques = [
    AllYes(),
    AllNo(),
    AllRandom()
]
tech_names = [t.get_name() for t in techniques]

results = ['\t'.join(('source', 'relation', 'target', 'gt', 'type', *tech_names))+'\n']
with open(dataset_file, 'r') as file:
    for line in file.readlines():
        source, relation, target, gt, n_type = line.split('\t')
        n_type = n_type.strip()
        tech_results = [str(t.score(source, relation, target)) for t in techniques]
        new_line = '\t'.join((source, relation, target, gt, n_type, *tech_results))+'\n'
        results.append(new_line)
with open(output_file, 'w') as file:
    file.writelines(results)