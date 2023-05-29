import typing
if typing.TYPE_CHECKING:
	from DataGen import KGDataset
from tqdm import tqdm, trange
from scipy.stats import ks_2samp
from math import floor

class Splitter():
	"""Interface used to split knowledge graphs"""
	def setKG(self, kg:"KGDataset"):
		self.kg = kg
	def initialize(self):
		raise NotImplementedError("This function is not implemented in the base class. Please, use other classes that extend it")
	def split(self)->dict[int,dict[str,dict[str,set]]]:
		"""
		Splits the graph using the parameters specified in the constructor.

		Returns: a dictionary with keys representing different split indices (only 0 if there are not several splits) and values representing the splits themselves.
		One split is a dictionary where the keys are strings representing the "training" and "testing" splits and the values are dictionaries with the triples of each split.
		The keys of each of these dictionaries are strings representing the "positive" and "negative" triples, with negatives being empty.
		"""

		raise NotImplementedError("This function is not implemented in the base class. Please, use other classes that extend it")

from random import random

class StatistitalSplitter(Splitter):
	"""Statistical splitter. Creates a single split while keeping a similar topology by using statistical methods"""

	def __init__(self, threshold=0.05):
		"""
		Splits the graph into training and testing sets.

		Arguments:
		threshold -- p-value threshold employed to compare node degrees distributions. Used to tune training/test sets size. 

		Generates and returns:
		graphs -- a dictionary with the training and testing sets as values in a dictionary with "train" and "test" keys. Both "train" and "test" return a dictionary with, so far, only the "positive" key corresponding to the positiva edges in a set.
		"""

		self.threshold = threshold
	
	def split(self):
		print("Performing split based on downscaling with statistical guarantees algorithm")

		graphs = dict()

		# We initialise the variable where the splits are stored:
		graphs[0] = dict()
		graphs[0]["train"] = dict()
		graphs[0]["test"] = dict()
		graphs[0]["valid"] = dict()
		graphs[0]["train"]["positive"] = self.kg.edges.copy()
		graphs[0]["test"]["positive"] = set()
		graphs[0]["train"]["negative"] = set()
		graphs[0]["test"]["negative"] = set()
		

		# We take a fraction of the edges of each relation
		for rel in tqdm(self.kg.relations):

			# We define an entity degree counter generator
			dg_gen = {entity:0 for entity in self.kg.entities.keys()}
			# We define a dictionary to store in and out degrees of nodes of the original graph and the new training set generated
			degrees = {"original_graph": {"in": dg_gen, "out": dg_gen}, "train": {"in": dg_gen, "out": dg_gen}}

			# We take the edges of the current relation
			edges = [(rel, s, t, 'P') for s, t in self.kg.grouped_edges[rel]]

			# For each entity of the graph we get its degrees for the current relation
			# (number of edges of this relation where the entity appears)
			for relation, s, t, p in edges:
				degrees["original_graph"]["out"][s] = degrees["original_graph"]["out"][s] + 1
				degrees["original_graph"]["in"][t] = degrees["original_graph"]["in"][t] + 1
				degrees["train"]["out"][s] = degrees["train"]["out"][s] + 1
				degrees["train"]["in"][t] = degrees["train"]["in"][t] + 1

			# For each edge, we aim to decide whether or not the triple can be safely discarded from the train set
			for relation, s, t, p in tqdm(edges):

				# We remove the edge from the train set
				graphs[0]["train"]["positive"].remove((relation,s,t))

				# Retrieve remaining entities in train set
				remaining_entities = set()
				for _, sr, tg in graphs[0]["train"]["positive"]:
					remaining_entities.update((sr, tg))

				# We check that, if edge is discarded, entities s and o will still be present in the new training split
				if (s in remaining_entities) and (t in remaining_entities):
					
					# We training set degrees dictionary
					degrees["train"]["out"][s] = degrees["train"]["out"][s] - 1
					degrees["train"]["in"][t] = degrees["train"]["in"][t] - 1

					# We check whether or not new train set degrees and original graph degrees distributions are statistically similar
					out_p_value = ks_2samp(list(degrees["original_graph"]["out"].values()), list(degrees["train"]["out"].values())).pvalue
					in_p_value = ks_2samp(list(degrees["original_graph"]["in"].values()), list(degrees["train"]["in"].values())).pvalue

					# To do so, we compare p-value with the given threshold
					if (out_p_value > self.threshold) and (in_p_value > self.threshold):
						# If the condition is met, we add the triple to the test set
						graphs[0]["test"]["positive"].add((relation,s,t))
					else:
						# Otherwise, we undo the changes made
						graphs[0]["train"]["positive"].add((relation,s,t))
						degrees["train"]["out"][s] = degrees["train"]["out"][s] + 1
						degrees["train"]["in"][t] = degrees["train"]["in"][t] + 1
				else:
					graphs[0]["train"]["positive"].add((relation,s,t))
		
		# Add positive symbol to triples in generated sets
		graphs[0]["train"]["positive"] = set([(r,s,t,'P') for (r,s,t) in graphs[0]["train"]["positive"]])
		graphs[0]["test"]["positive"] = set([(r,s,t,'P') for (r,s,t) in graphs[0]["test"]["positive"]])

		# Print sets sizes and split percentages
		train_percentage = round(len(graphs[0]["train"]["positive"])/(len(graphs[0]["train"]["positive"])+len(graphs[0]["test"]["positive"]))*100,2)
		test_percentage = round(len(graphs[0]["test"]["positive"])/(len(graphs[0]["train"]["positive"])+len(graphs[0]["test"]["positive"]))*100,2)
		print("P-value threshold:", self.threshold, "\nTraining triples:", len(graphs[0]["train"]["positive"]), "("+str(train_percentage)+"%)", "\nTest triples:", len(graphs[0]["test"]["positive"]), "("+str(test_percentage)+"%)",)

		return graphs
		
class RandomSplitter(Splitter):
	"""Random splitter. Can create several splits according to different random separations"""

	def __init__(self, fraction_test=0.1, fraction_test_relations={}, number_splits=1):
		"""
		Splits the graph into training and testing sets. Creates as many different splits as given by the "number_splits" property

		Arguments:
		fraction_test_relations -- a dictionary with the fraction of each relation to take for testing. Default: an empty dictionary, which implies the same fraction for all relations (given by "fraction_test")
		fraction_test -- the fraction to take from all relations for testing. Only used if "fraction_test_relations" is empty.

		Generates and returns:
		graphs -- a dictionary with the split identifier as keys, and the training and testing sets of each split as values in a dictionary with "train" and "test" keys. Both "train" and "test" return a dictionary with, so far, only the "positive" key corresponding to the positiva edges in a set.
		"""

		self.fraction_test = fraction_test
		self.fraction_test_relations = fraction_test_relations
		self.number_splits = number_splits

	def split(self):

		print("Performing randomnly selected edges splitting")
		graphs = dict()
		# If we do not provide a fraction of testing for each individual relation, the same fraction is used for all of them
		if(len(self.fraction_test_relations) == 0):
			self.fraction_test_relations = {rel: self.fraction_test for rel in self.kg.relations}
		# We create a variable number of splits.
		for i in trange(self.number_splits):
			# We initialise the variable where the splits are stored
			graphs[i] = dict()
			graphs[i]["train"] = dict()
			graphs[i]["test"] = dict()
			graphs[i]["train"]["positive"] = set()
			graphs[i]["test"]["positive"] = set()
			graphs[i]["train"]["negative"] = set()
			graphs[i]["test"]["negative"] = set()
			# We take a fraction of the edges of each relation
			for rel in tqdm(self.kg.relations):
				# We take the edges of the current relation
				edges = [(rel, s, t, 'P') for s, t in self.kg.grouped_edges[rel]]
				# Different splits use different offsets for the edges that are taken for testing
				offset = floor(len(edges) / self.number_splits * i)
				# We take the fraction for the current relation
				fraction_test = self.fraction_test_relations.get(rel, 0.0)
				# We compute how many edges will be taken for testing
				num_test = floor(len(edges) * fraction_test)
				# We compute what will be the indices of the edges that will be taken for training and testing, using the offset
				ids_test = [(offset + x) % len(edges) for x in range(0, num_test)]
				ids_train = [(offset + x) % len(edges) for x in range(num_test, len(edges))]
				# We take the edges of each set using the indices
				edges_test = [edges[id] for id in ids_test]
				edges_train = [edges[id] for id in ids_train]
				# Finally, we store them in the variable
				graphs[i]["test"]["positive"].update(edges_test)
				graphs[i]["train"]["positive"].update(edges_train)
		
		return graphs