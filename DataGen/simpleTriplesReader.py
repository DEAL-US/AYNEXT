from reader import Reader
from random import random

class SimpleTriplesReader(Reader):
	"""Reader of simple triples files with one line per triple. Assumes the order is <relation, source, target>"""

	def __init__(self, file_path, separator, prob):
		"""
		Arguments:

		file_path -- the path to the single file containing the knwoledge graphs
		separator -- the separatior character or string used to separate the elements of the triple
		prob -- probability of keeping each triple when reading the graph. 
		If 1.0, the entire graph is kept. If lesser than one, the final graph has reduced size.
		"""

		self.file_path = file_path
		self.separator = separator
		self.prob = prob

	def read(self):
		"""
		Reads the graph using the parameters specified in the constructor.
		Expects each line to contain a triple with the relation first, then the source, then the target.

		Returns: a tuple with:
		1: a dictionary with the entities as keys (their names) as degree information as values.
		Each value is a dictionary with the outwards degree ("out_degree key"), inwards degree ("in_degree key"), and total degree ("degree" key).
		2: a set with the name of the relations in the graph
		3: a set with the edges in the graph. Each edge is a tuple with the name of the relation, the source entity, and the target entity.
		"""

		entities = dict()
		relations = set()
		edges = set()

		with open(self.file_path, "r") as file:
			for line in file:
				if(random() < self.prob):
					line = line.strip()
					triple = line.split(self.separator)
					source = triple[0]
					relationship = triple[1]
					target = triple[2]

					# Adding entities, relations and edges
					if source not in entities:
						entities[source] = dict(degree=0, out_degree=0, in_degree=0, data_properties={})
					if target not in entities:
						entities[target] = dict(degree=0, out_degree=0, in_degree=0, data_properties={})
					entities[source]["out_degree"] += 1
					entities[target]["in_degree"] += 1
					entities[source]["degree"] += 1
					entities[target]["degree"] += 1

					relations.add(relationship)
					edges.add((relationship, source, target))
		return (entities, relations, edges)