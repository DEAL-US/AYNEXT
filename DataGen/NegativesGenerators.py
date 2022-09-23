import typing
if typing.TYPE_CHECKING:
	from DataGen import KGDataset
import numpy as np
from tqdm import tqdm
from random import choice

class NegativesGenerator():
	"""Interface used to generate negative examples"""
	def setKG(self, kg:"KGDataset"):
		self.kg = kg
	def initialize(self):
		raise NotImplementedError("This function is not implemented in the base class. Please, use other classes that extend it")
	def generate_negatives(self, positive, number_negatives):
		raise NotImplementedError("This function is not implemented in the base class. Please, use other classes that extend it")

class PPRGenerator(NegativesGenerator):
	def __init__(self, steps:int=5, alpha:float=0.02) -> None:
		"""
		Creates the generator object

		Arguments:
		steps -- the number of steps of the random walks used to compute PPR. If None, defaults to 1/alpha.
		alpha -- the teleport probability during the random walks. Increase to focus probability around each source node. Default: 0.02
		"""

		self.steps = steps
		self.alpha = alpha
		self.tc = "CB"

	def initialize(self):
		"""
		Computes the personalised page rank of every entity, using only outward edges for paths.
		
		Generates and stores:
		ranks -- a matrix of size NxN where N is the number of entities. Position i,j corresponds to the probability of reaching entity j from entity i after a random walk of the given number of steps and the given teleport probability during each step.
		"""

		if(not self.kg.has_encoding):
			self.kg.encode(False)

		print("Computing movements matrix")
		matrix_movements = np.empty([len(self.entities), len(self.entities)])
		for entity in tqdm(self.kg.entities):
			source_ind = self.kg.etoint[entity]
			edges = self.kg.entity_edges.get(entity, list())
			entity_count = dict()
			# We compute how many edges there are towards each node
			for edge in edges:
				target = edge[2]
				entity_count[target] = entity_count.get(target, 0) + 1
			for target, frequency in entity_count.items():
				target_ind = self.kg.etoint[target]
				# Since all walks are equally probable, the probability is always the number of edges that go to the target node divided by the total number of outgoing edges of the node
				matrix_movements[source_ind, target_ind] = frequency / len(edges)
		print("Computing PPR for every entity")
		# Steps defaults to 1/alpha
		if steps is None:
			steps = round(1 / self.alpha)
		initial_distribution = np.identity(len(self.kg.entities))
		# The initial matrix represents each node as a starting point, that is, 0 steps
		ranks = np.identity(len(self.kg.entities))
		# Each matrix multiplication represents a step
		for _ in tqdm(range(steps)):
			ranks = (1 - self.alpha) * np.matmul(ranks, matrix_movements) + self.alpha * initial_distribution
		self.ranks = ranks

	def generate_negatives(self, positive, number_negatives):
		"""
		Generates negatives from a positive using the PPR strategy, which changes the source and target while keeping the domain/range of the relation.
		The candidates are selected from the PPR of each node, selecting a random one while weighting by PPR.

		Arguments:
		positive -- the positive to generate the negatives from
		number_negatives -- how many negatives to generate

		Returns: a list of negative edge examples.
		"""
		rel = positive[0]
		# We take the rank of each existing entity for both the source and target node
		source_ranks = self.ranks[self.kg.etoint[positive[1]]]
		target_ranks = self.ranks[self.kg.etoint[positive[2]]]
		# We take as candidates for the change the entities with a rank above 0, and which are in the range/domain of the relation
		sources = [(self.kg.inttoe[i], rank) for i, rank in enumerate(source_ranks) if rank > 0 and self.kg.inttoe[i] in self.kg.domains[rel]]
		targets = [(self.kg.inttoe[i], rank) for i, rank in enumerate(target_ranks) if rank > 0 and self.kg.inttoe[i] in self.kg.ranges[rel]]
		# We turn the ranks into arrays of probabilities by dividing them by the sum of the original array
		sources_probs = np.array([source[1] for source in sources])
		sources_probs /= sources_probs.sum()
		targets_probs = np.array([target[1] for target in targets])
		targets_probs /= targets_probs.sum()
		# We use the probabilities as weights to select new sources and targets
		ids_sources = np.random.choice(len(sources_probs), number_negatives, p=sources_probs)
		ids_targets = np.random.choice(len(targets_probs), number_negatives, p=targets_probs)
		# We select the sources and targets, and create the negatives
		sources = [sources[ids_sources[i]][0] for i in range(number_negatives)]
		targets = [targets[ids_targets[i]][0] for i in range(number_negatives)]
		negatives = [(rel, sources[i], targets[i], self.tc) for i in range(number_negatives)]
		return negatives

class RandomGenerator(NegativesGenerator):
	def __init__(self, keep_dom_ran=True, change_source=False, change_target=True, equal_probabilities=False) -> None:
		'''
		Arguments:
		keep_dom_range -- whether or not to keep the domain or range when finding candidates. Default: True.
		change_source -- whether or not to change the source when generating negative examples. Default: False.
		change_target -- whether or not to change the target when generating negative examples. Default: True.
		equal_probabilities -- whether or not to give the same probability to all candidates. If False, the probability depends on the number of occurrences of each entity in the relevant position of the relation. Default: False.
		'''
		
		self.keep_dom_ran = keep_dom_ran
		self.change_source = change_source
		self.change_target = change_target
		self.equal_probabilities = equal_probabilities
		if(change_source and change_target):
			self.tc = "CB"
		elif(change_source):
			self.tc = "CS"
		elif(change_target):
			self.tc = "CT"
		self.candidates_cache = {source_target: {} for source_target in ("source", "target")}

	def initialize(self):
		pass

	def get_candidates(self, relation, source_target):
		if(relation in self.candidates_cache[source_target]):
			candidates = self.candidates_cache[source_target][relation]
		else:
			if(source_target == "source"):
				candidates = [edge[0] for edge in self.kg.grouped_edges[relation]]
			else:
				candidates = [edge[1] for edge in self.kg.grouped_edges[relation]]
			self.candidates_cache[source_target][relation] = candidates
		return candidates
	
	def generate_negatives(self, positive, number_negatives):
		"""
		Generates negatives from a positive by changing the source and/or target.

		Arguments:
		positive -- the positive to generate the negatives from.
		number_negatives -- how many negatives to generate.
		
		Returns: a list of negative edge examples.
		"""

		rel = positive[0]
			
		# If we keep the domain and range, the candidates must be taken from the edges of the same relation as the positive
		if(self.keep_dom_ran):
			if(self.change_source):
				candidates_source = self.get_candidates(rel, "source")
			if(self.change_target):
				candidates_target = self.get_candidates(rel, "target")
		# Otherwise, they are taken from all edges
		else:
			if(self.change_source):
				candidates_source = self.kg.ents_source
			if(self.change_target):
				candidates_target = self.kg.ents_target
		# If every candidate is equally probable, without taking their frequency of appearance in the edges into account, we use the domain and range of each relation, where entities do not appear twice in each list
		if(self.equal_probabilities):
			if(self.change_source):
				candidates_source = list(self.kg.domains[rel])
			if(self.change_target):
				candidates_target = list(self.kg.ranges[rel])
		negatives = list()
		# Loop for each generated negative. We do not generate all of them at once, in order to check failed attempts to generate negatives
		for _ in range(number_negatives):
			source = None
			target = None
			# We find a new source, if required
			if(self.change_source):
				attempts = 0
				found = False
				# We only try to find a new source in 20 attempts. We assume it is not possible if surpassed (all candidates are equal to the original, and thus we can not change it)
				while not found and len(candidates_source) > 1 and attempts <= 20:
					# After 10 attempts, it is difficult to find a new source. The original could be very frequent. We make all probabilities equal
					if(attempts > 10):
						if(self.keep_dom_ran):
							candidates_source = list(self.kg.domains[rel])
						else:
							candidates_source = list(self.kg.entities.keys())
					attempts += 1
					# We take the new source among the candidates
					source = choice(candidates_source)
					# The attempt is only successful if the new source is different from the original one
					found = source != positive[1]
					if not found:
						source = None
			# If not required, the source remains the same
			else:
				source = positive[1]
			# Finding a new target, if required. The process is the same as with the source
			if(self.change_target):
				attempts = 0
				found = False
				while not found and len(candidates_target) > 1 and attempts <= 20:
					if(attempts > 10):
						if(self.keep_dom_ran):
							candidates_target = list(self.kg.ranges[rel])
						else:
							candidates_target = list(self.kg.entities.keys())
					attempts += 1
					target = choice(candidates_target)
					found = target != positive[2]
					if not found:
						target = None
			else:
				target = positive[2]
			# We only add the negative if both the source and target changes, if required, were successful
			if(not (self.change_source and source is None) and not (self.change_target and target is None)):
				negatives.append((rel, source, target, self.tc))
		return negatives