from dataclasses import replace
from itertools import combinations
from math import floor
import networkx as nx
from random import random
import os
from tqdm import tqdm, trange
from bokeh.plotting import figure
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import dodge
from bokeh.embed import components
import json
import datetime
from Readers import SimpleTriplesReader, NTriplesReader, LinkedDataReader
from NegativesGenerators import PPRGenerator, RandomGenerator, NegativesGenerator
from Splitters import RandomSplitter, StatistitalSplitter
import argparse
from scipy.stats import ks_2samp

"""
This script generates evaluation datasets for knowledge graph completion techniques.
The main function can be found at the end of the file. Use the --help command to obtain a description of the arguments.
"""

VERSION = "1.5.0"
# html imports for the generated html summary
bokeh_js_import = '''<link
    href="https://cdn.pydata.org/bokeh/release/bokeh-2.4.2.min.css"
    rel="stylesheet" type="text/css">
<link
    href="https://cdn.pydata.org/bokeh/release/bokeh-widgets-2.4.2.min.css"
    rel="stylesheet" type="text/css">
<link
    href="https://cdn.pydata.org/bokeh/release/bokeh-tables-2.4.2.min.css"
    rel="stylesheet" type="text/css">
<link href="https://netdna.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" rel="stylesheet"/>

<script src="https://cdn.pydata.org/bokeh/release/bokeh-2.4.2.min.js"></script>
<script src="https://cdn.pydata.org/bokeh/release/bokeh-widgets-2.4.2.min.js"></script>
<script src="https://cdn.pydata.org/bokeh/release/bokeh-tables-2.4.2.min.js"></script>'''

class KGDataset():
	"""
	Class used to represent the datasets.

	Methods that should be used externally:
	read -- Preprocessing: reads the knowledge graph using a reader, and performs preprocessing.
	split -- Splitting: generates training and testing sets.
	generate_negatives -- Negatives generation: adds negative examples to the training and testing sets.
	export_files -- Generates the output files excluding the gexf file.
	export_gexf -- Generates a gexf file with the evaluation datasets.
	compute_PPR -- computes the personalised page rank for each entity in the graph.
	"""

	def __init__(self, results_directory):
		"""Arguments:

		results_directory -- the output spliter.
		number splits -- in case several training/set splits must be generated. Default: 1 split, corresponding to one training/set split
		"""
		self.has_encoding = False
		self.ents_source = []
		self.ents_target = []
		self.entities = dict()
		self.relations = set()
		self.edges = set()
		self.inverses = set()
		self.inverse_tuples = list()
		self.inverses_dict = dict()
		self.graphs = dict()
		self.results_directory = results_directory
		self.entity_edges = dict()
		self.domains = dict()
		self.ranges = dict()
		if not os.path.exists(results_directory):
			os.makedirs(results_directory)

	def group_edges(self):
		"""
		Groups the edges in a per relation basis.
		Creates and stores a dictionary with the relations as keys and the set of edges of each relation as values.
		"""

		print("\nGrouping edges")
		self.grouped_edges = dict()
		self.domains = dict()
		self.ranges = dict()
		for edge in self.edges:
			if edge[0] not in self.grouped_edges:
				self.grouped_edges[edge[0]] = set()
			if edge[0] not in self.domains:
				self.domains[edge[0]] = set()
			if edge[0] not in self.ranges:
				self.ranges[edge[0]] = set()
			self.grouped_edges[edge[0]].add((edge[1], edge[2]))
			self.domains[edge[0]].add(edge[1])
			self.ranges[edge[0]].add(edge[2])
			self.ents_source.append(edge[1])
			self.ents_target.append(edge[2])

	def is_type(self, relation):
		"""
		Determines whether or not a given relation denotes the type of entities.

		Arguments:
		relation -- the name of the relation to be checked.

		Returns:
		True if relation is used to denote types, False otherwise.
		"""
		# So far, we only check that the relation has a typical type name
		return "rdf-syntax-ns#type" in relation

	def read(self, reader, inverse_threshold=0.9, min_num_rel=0, reach_fraction=1, remove_inverses=False, create_summary=True, separate_types=False):
		"""
		Reads the knowledge graph using a reader, and performs preprocessing. This function corresponds to the preprocessing step of the workflow.

		Arguments:
		min_num_rel -- minimum frequency required to keep a relation. Default: 0 (keep all).
		reach_fraction -- fraction of the total number of edges to keep, accumulating the relations, sorted by frequency. Default: 1 (keep all).
		remove_inverses -- whether or not to remove relations detected as inverses. Default: False.
		create_summary -- whether or not to create an html summary including tables and plots with the frequency of each relation and degree of each entity. Note: inverses are always included in this summary, even if removed. Default: True.
		separate_types -- whether or not the types of entities (contained in triples) should be separated from the graph and set apart

		Generates and stores:
		entities -- a dictionary with the entities as keys (their names) and a dictionary with information about each entity as values.
		Each value is a dictionary with the outwards degree ("out_degree key"), inwards degree ("in_degree key"), total degree ("degree" key), data properties ("data_properties" key), and types ("types" key).
		relations -- a set with the name of the relations in the graph
		edges -- a set with the edges in the graph. Each edge is a tuple with the name of the relation, the source entity, and the target entity.
		the inverses as detailed in function find_inverses
		entity_edges -- a dictionary with the entities as keys, and the set of their outgoing edges as values
		"""

		print("\nReading graph")
		self.entities, self.relations, self.edges = reader.read()
		self.group_edges()

		# If we separate the types...
		if(separate_types):
			type_rels = list()
			# We first associate an empty list of types to each entity
			for entity in self.entities:
				self.entities[entity]["types"] = set()
			print("\nStoring entity types")
			# We iterate each relation
			for rel, instances in self.grouped_edges.items():
				# If it denotes a type, we add the types denoted by its instances and add it to a list of type relations
				if(self.is_type(rel)):
					type_rels.append(rel)
					for source, target in instances:
						self.entities[source]["types"].add(target)
			print("\nRemoving type relations")
			# Finally, we remove the type relations from the graph
			self.remove_rels(type_rels)

		print("\nPruning relations")

		# First filtering of relations by selecting only those with a frequency above the frequency threshold
		candidate_rels = [(rel, len(instances)) for rel, instances in self.grouped_edges.items() if len(instances) >= min_num_rel]
		# Sorting of the candidates by frequency, in order to compute the accumulated fraction
		candidate_rels.sort(key=lambda x: x[1], reverse=True)
		accepted_rels = list()
		# The list of amounts (frequencies) and accumulated fractions will be used to generate the visual summary
		amounts = list()
		accumulated_fractions = list()
		accumulated_fraction = 0.0
		y_values = list()
		with tqdm(total=len(self.edges)) as pbar:
			for rel, amount in candidate_rels:
				# We add the relations and update the variables that store data about them
				accepted_rels.append(rel)
				amounts.append(amount)
				accumulated_fraction += amount / len(self.edges)
				accumulated_fractions.append(accumulated_fraction)
				y_values.append(accumulated_fraction)
				pbar.update(amount)
				pbar.refresh()
				# If the threshold is reached, stop adding relations
				if accumulated_fraction >= reach_fraction:
					break

		print(f'Kept {len(accepted_rels)} relations out of {len(self.relations)}')
		# We remove all rels that are not in the set of accepted ones
		removed_rels = [rel for rel in self.relations if rel not in accepted_rels]
		print("\nRemoving small relations")
		self.remove_rels(removed_rels)
		# We create the visual summary
		if(create_summary):
			self.create_summary(accepted_rels, amounts, accumulated_fractions)
		# We find inverses, and remove them if necessary
		self.find_inverses(inverse_threshold)
		if(remove_inverses):
			print("\nRemoving inverses")
			self.remove_rels(self.inverses)
		# We explicitely store the outgoing edges of each entity node
		print("Storing outgoing edges for each node")
		for edge in tqdm(self.edges):
			if(edge[1] not in self.entity_edges):
				self.entity_edges[edge[1]] = list()
			self.entity_edges[edge[1]].append(edge)

	def create_summary(self, relations, amounts, accumulated_fractions):
		"""
		Creates the html summary of the relation frequencies and entity degrees

		Arguments:
		relations -- a list with the relations to include in the summary.
		amounts -- a list with the frequency of each relation, in the same order as "relations".
		accumulated_fractions -- a list with the accumulated fraction of each relation, in the same order as "relations".
		"""

		print("Creating summary")

		source_relations = ColumnDataSource(data=dict(x=relations, frequencies=amounts, accumulated_fractions=accumulated_fractions))
		source_relations_table = ColumnDataSource(data=dict(x=relations, frequencies=amounts, accumulated_fractions=accumulated_fractions))
		print(relations[:5])
		p = figure(x_range=relations, plot_height=350, title="Relation frequency histogram")
		p.vbar(x="x", top="frequencies", width=0.9, source=source_relations)
		p.xgrid.grid_line_color = None
		p.y_range.start = 0
		p.add_tools(HoverTool(tooltips=[("Relation", "@x"), ("Frequency", "@frequencies")]))
		p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
		p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
		p.xaxis.major_label_text_font_size = '0pt'  # turn off x-axis tick labels
		relations_script, relations_div = components(p)
		columns = [
			TableColumn(field="x", title="Relation name"),
			TableColumn(field="frequencies", title="Frequency"),
			TableColumn(field="accumulated_fractions", title="Accumulated fraction")
		]
		data_table = DataTable(source=source_relations_table, columns=columns, width=450, height=350)
		relations_table_script, relations_table_div = components(data_table)

		entities = sorted(self.entities.items(), key=lambda x: x[1]["degree"], reverse=True)
		source_entities = ColumnDataSource(	data=dict(x=[entity[0] for entity in entities], 
											degree=[entity[1]["degree"] for entity in entities], 
											out_degree=[entity[1]["out_degree"] for entity in entities], 
											in_degree=[entity[1]["in_degree"] for entity in entities]))
		source_entities_table = ColumnDataSource(	data=dict(x=[entity[0] for entity in entities], 
													degree=[entity[1]["degree"] for entity in entities], 
													out_degree=[entity[1]["out_degree"] for entity in entities], 
													in_degree=[entity[1]["in_degree"] for entity in entities]))
		p = figure(x_range=[entity[0] for entity in entities], plot_height=350, title="Entity degree histogram")
		p.vbar(color="#c9d9d3", x=dodge('x', -0.25, range=p.x_range), top="degree", width=0.2, source=source_entities)
		p.vbar(color="#718dbf", x=dodge('x', 0, range=p.x_range), top="out_degree", width=0.2, source=source_entities)
		p.vbar(color="#e84d60", x=dodge('x', 0.25, range=p.x_range), top="in_degree", width=0.2, source=source_entities)
		p.xgrid.grid_line_color = None
		p.y_range.start = 0
		p.add_tools(HoverTool(tooltips=[("Entity", "@x"), ("Degree", "@degree"), ("Outwards degree", "@out_degree"), ("Inwards degree", "@in_degree")]))
		p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
		p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
		p.xaxis.major_label_text_font_size = '0pt'  # turn off x-axis tick labels
		entities_script, entities_div = components(p)
		columns = [
			TableColumn(field="x", title="Entity name"),
			TableColumn(field="degree", title="Total degree"),
			TableColumn(field="out_degree", title="Outwards degree"),
			TableColumn(field="in_degree", title="Inwards degree")
		]
		data_table = DataTable(source=source_entities_table, columns=columns, width=450, height=350)
		entities_table_script, entities_table_div = components(data_table)
		with open(self.results_directory + "/summary.html", "w") as file:
			file.write(f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>AYNEC graph summary</title>
				{bokeh_js_import}

				{relations_script}
				{relations_table_script}
				{entities_script}
				{entities_table_script}

			</head><body><section class="container"><h1>AYNEC summary - {self.results_directory.split('/')[-1]}</h1><h2>Relations</h2><div class="row"><div class="col-md-7">
				{relations_div}
			</div><div class="col-md-5">
				{relations_table_div}
			</div></div><hb/><h2>Entities</h2><div class="row"><div class="col-md-7">
				{entities_div}
			</div><div class="col-md-5">
				{entities_table_div}
			</div></div></section>
			<section style="margin-top:20px" class="container"><div class="row text-center"><div class="col-md-12">Generated with AYNEC {VERSION} at {datetime.datetime.now()}. For issues or suggerences send a mail to <a href="mailto:dayala@us.es?Subject=AYNEC%20issue" target="_top">dayala1@us.es</a></div></div></section>
			</body></html>''')

	def remove_rels(self, removed_rels):
		"""
		Removes the given relations form the stored graph.

		Arguments:
		removed_rels -- the relations to be removed.
		"""
		self.edges = set(filter(lambda e: e[0] not in removed_rels, self.edges))
		with tqdm(removed_rels) as pbar:
			for rel in pbar:
				# pbar.write(rel)
				self.grouped_edges.pop(rel, None)
				self.relations.remove(rel)

	def find_inverses(self, overlapping_threshold):
		"""
		Finds the inverse relations in the graph.

		Arguments:
		overlapping_threshold -- the fraction of edges that must have an inverse in the other relation to consider the pair of relations as inverses.

		Computes and stores:
		inverse_tuples -- a list of tuples with each inverse, where each tuple contains the two relations in the inverse relationship.
		inverses -- a set with the inverses, that is, the second element of the inverse tuples.
		inverses_dict -- a dictionary with the relations as keys and the sets of their inverse relations as values.

		"""

		print("\nFinding inverse relations")
		# Every pair of relations (without order) is a possible pair of inverses
		for combination in tqdm(combinations(self.relations, 2), total=len(self.relations) * (len(self.relations) - 1) / 2):
			edges1 = self.grouped_edges[combination[0]]
			edges2 = self.grouped_edges[combination[1]]
			# We compute the fraction of edges from each relation that are found in the other relation, inversed
			inversed_1_to_2 = [(edge[1], edge[0]) in edges2 for edge in edges1].count(True) / len(edges1)
			inversed_2_to_1 = [(edge[1], edge[0]) in edges1 for edge in edges2].count(True) / len(edges2)
			# If both fractions are above the threshold, they are considered to be a pair of inverses
			is_inverse = (inversed_1_to_2 > overlapping_threshold) and (inversed_2_to_1 > overlapping_threshold)
			# The pair is added to a list of tuples. The smallest relation is also added to a list of inverses, which are removed if the relevant option is toggled
			if is_inverse:
				self.inverse_tuples.append((combination[0], combination[1]))
				if(len(edges1) > len(edges2)):
					self.inverses.add(combination[1])
				else:
					self.inverses.add(combination[0])
		print(f'found {len(self.inverse_tuples)} inverses')
		# We also store in a dictionary the set of inverses of each relation
		for r1 in self.relations:
			self.inverses_dict[r1] = set()
			for r2 in self.relations:
				if (r1, r2) or (r2, r1) in self.inverse_tuples:
					self.inverses_dict[r1].add(r2)

	def encode(self, replace_names=False):
		"""
		Associates relations and entities to contiguous integers.

		Arguments:
		replace_names -- whether or not to, in addition to storing the mapping to integers, to actually replace the name of the stored entities and relations with said integers. Default: False.

		Computes and stores:
		etoint -- the dictionary with entities as keys and integers as values
		inttoe -- the dictionary with integers as keys and entities as values
		rtoint -- the dictionary with relations as keys and integers as values
		inttor -- the dictionary with integers as keys and relations as values

		"""
		self.has_encoding = True
		self.etoint = {ent: i for i, ent in enumerate(self.entities.keys())}
		self.inttoe = {i: ent for ent, i in self.etoint.items()}
		self.rtoint = {rel: i for i, rel in enumerate(self.relations)}
		self.inttor = {i: rel for rel, i in self.rtoint.items()}
		if(replace_names):
			self.edges = [(self.etoint[s], self.rtoint[r], self.etoint[t]) for r, s, t in self.edges]
			self.relations = [self.rtoint[r] for r in self.relations]
			for e in self.entities.keys():
				self.entities[self.etoint[e]] = self.entities.pop(e)

			self.group_edges()

	def add_networkx_edges(self, split, train_test, positive_negative, entities, graph):
		"""
		Adds edges to a networkx graph.

		Arguments:
		split -- the split to add edges from.
		train_test -- whether to add the train or test edges. Should be "train" or "test".
		positive_negative -- whether to add the positive or the negative examples. Should be "positive" or "negative".
		entities -- a set with entities, used to keep track of the entities that are being added to a single graph.
		"""

		edges = self.graphs[split][train_test][positive_negative]
		entities.update([str(edge[1]) for edge in edges])
		entities.update([str(edge[2]) for edge in edges])
		graph_edges = [(str(edge[1]), str(edge[2]), {"Label": str(edge[0]), "positive": True if positive_negative == "positive" else False, "train": True if train_test == "train" else False, "type": edge[3]}) for edge in edges]
		print(f'Adding {len(graph_edges)} edges')
		graph.add_edges_from(graph_edges)

	def export_gexf(self, split, include_train, include_test, include_positive, include_negative):
		"""
		Generates and stores the gexf file in the output folder, named "dataset.gexf".

		Arguments:
		split -- the split to use as source of the graph
		include_train -- whether or not to include the training edges
		include_test -- whether or not to include the testing edges
		include_train -- whether or not to include the positive edges
		include_test -- whether or not to include the negative edges
		"""

		print("\nExporting gexf")
		g = nx.MultiDiGraph()
		entities = set()
		if(include_train):
			if(include_positive):
				self.add_networkx_edges(split, "train", "positive", entities, g)
			if(include_negative):
				self.add_networkx_edges(split, "train", "negative", entities, g)
		if(include_test):
			if(include_positive):
				self.add_networkx_edges(split, "test", "positive", entities, g)
			if(include_negative):
				self.add_networkx_edges(split, "test", "negative", entities, g)
		g.add_nodes_from(entities)
		nx.write_gexf(g, self.results_directory + "/dataset.gexf")

	def get_candidates(self, relation, source_target):
		if(relation in self.candidates_cache[source_target]):
			candidates = self.candidates_cache[source_target][relation]
		else:
			if(source_target == "source"):
				candidates = [edge[0] for edge in self.grouped_edges[relation]]
			else:
				candidates = [edge[1] for edge in self.grouped_edges[relation]]
			self.candidates_cache[source_target][relation] = candidates
		return candidates

	def generate_negatives(self, split:int, train_test:str, generators_and_number:"dict[NegativesGenerator,]", clean_before=False, reject_rel_after_failure=False):
		"""
		Generates negatives from a given set of positive examples, adding in parameter "tp" whether the source (CS),
		the target (CT) or both (CB) elements of the positive were corrupted.

		Arguments:
		split -- the split form which to generate negatives
		train_test -- whether to generate negatives from the training or testing set
		generators_and_number -- a dictionary containing negatives negerators as keys and the numbers of negatives to generate for each one as values
		clean_before -- whether or not remove existing negative examples for the given set, if there are any. Default: False
		reject_rel_after_failute -- whether or not ignore a relation if an attempt to generate negative examples form a positive of the relation is unable to find any, which should mean that there is only one candidate
		"""

		print("\nGenerating negatives")
		# We generate negatives for each positive example, be it in the training or the testing set, for a given split
		edges = self.graphs[split][train_test]["positive"]
		# If so specified, we remove the formerly stored negatives
		if clean_before:
			self.graphs[split][train_test]["negative"] = set()
		negatives = list()
		with tqdm(edges) as pbar:
			for positive in pbar:
					for generator, num_negatives in generators_and_number.items():
						ignored_rels_positives = set()
						# If a relation as been marked as "to be ignored", we skip the negatives generation. Relations are marked as such if it is impossible to generate negatives for a positive example
						if(positive[0] not in ignored_rels_positives):
							new_negatives = generator.generate_negatives(positive, num_negatives)
							# If the generation of negatives was succesful, we add them to the set of negatives
							if(len(new_negatives)) > 0:
								negatives.extend(new_negatives)
							# Otherwise, if the option is toggled, we mark the relation so that there are no more attempts to generate negatives from its instances
							elif(reject_rel_after_failure):
								ignored_rels_positives.add(positive[0])
								pbar.write(f'Ignoring relation {positive[0]}: returned no negatives in an attempt')
							pbar.refresh()
					
		self.graphs[split][train_test]["negative"] = negatives

	def create_validation(self, valid_fraction):
		# TODO Separar un porcentaje a validación. Cuidado con coger los negativos de cada positivo

	def export_files(self, split, include_train_negatives, include_dataproperties, include_types):
		"""
		Creates the output files, excluding the gexf files.

		Arguments:
		split -- the split to generate the output from.
		include_train_negatives -- whether or not training negatives should be included.

		Outputs the following files:
		train.txt -- the training triples, with a triple per line, separated by tabs and with a label in the following order: <source relation target label>. Label is 1 if positive and -1 if negative.
		test.txt -- the testing triples, following the same format as train.txt
		relations.txt -- the existing relations and their frequency, sorted by frequency.
		entities.txt -- the existing entities and their degrees, sorted by total degree.
		inverses.txt -- the detected inverse relations pairs, whether or not they were removed.
		data_properties.json -- the data properties of each entity
		types.json -- the types of each entity
		"""
		print("Exporting train triples")
		with open(self.results_directory + "/train.txt", "w", encoding="utf-8") as file:
			for edge in self.graphs[split]["train"]["positive"]:
				file.write("\t".join((edge[1], edge[0], edge[2], "1", edge[3])) + "\n")
			if(include_train_negatives):
				for edge in self.graphs[split]["train"]["negative"]:
					file.write("\t".join((edge[1], edge[0], edge[2], "-1", edge[3])) + "\n")
		print("Exporting test triples")
		with open(self.results_directory + "/test.txt", "w", encoding="utf-8") as file:
			for edge in self.graphs[split]["test"]["positive"]:
				file.write("\t".join((edge[1], edge[0], edge[2], "1", edge[3])) + "\n")
			for edge in self.graphs[split]["test"]["negative"]:
				file.write("\t".join((edge[1], edge[0], edge[2], "-1", edge[3])) + "\n")
		print("Exporting relations")
		with open(self.results_directory + "/relations.txt", "w", encoding="utf-8") as file:
			for rel, edges in sorted(self.grouped_edges.items(), key=lambda x: len(x[1]), reverse=True):
				file.write(f'{rel}\t{str(len(edges))}\n')
		print("Exporting entities")
		with open(self.results_directory + "/entities.txt", "w", encoding="utf-8") as file:
			for entity, degrees in sorted(self.entities.items(), key=lambda x: x[1]["degree"], reverse=True):
				file.write(f'{entity}\t{degrees["degree"]}\t{degrees["out_degree"]}\t{degrees["in_degree"]}\n')
		print("Exporting inverses")
		with open(self.results_directory + "/inverses.txt", "w", encoding="utf-8") as file:
			for r1, r2 in self.inverse_tuples:
				file.write(f'{r1}\t{r2}\n')
		if(include_dataproperties):
			print("Exporting data properties")
			data_properties = {entity: {data_property: value for data_property, value in properties["data_properties"].items()} for entity, properties in self.entities.items() if properties["data_properties"]}
			with open(self.results_directory + "/data_properties.json", "w", encoding="utf-8") as file:
				json.dump(data_properties, file)
		if(include_types):
			print("Exporting entity types")
			types = {entity: [type for type in properties["types"]] for entity, properties in self.entities.items() if properties["types"]}
			with open(self.results_directory + "/types.json", "w", encoding="utf-8") as file:
				json.dump(types, file)

def generate_datasets(	input_file,
				 		input_format, 
						output_folder, 
						graph_fraction, 
						generate_negatives_training, 
						remove_inverses, 
						min_num_rel, 
						reach_fraction, 
						testing_fraction, 
						validation_fraction,
						splitting_technique,
						pvalue_threshold,
						negatives_generators,
						export_gexf,
						create_summary,
						inverse_threshold,
						include_data_prop,
						separate_types):
	# We read and preprocess the graph
	if(input_format == "nt"):
		reader = NTriplesReader(input_file, graph_fraction, include_data_prop)
	elif(input_format in ["rdfa", "nt", "n3", "xml", "trix"]):
		reader = LinkedDataReader(input_file, graph_fraction, include_data_prop, input_format)
	else:
		reader = SimpleTriplesReader(input_file, '\t', graph_fraction)
	kgd = KGDataset(output_folder)
	print(output_folder)
	kgd.read(reader, inverse_threshold=inverse_threshold, min_num_rel=min_num_rel, reach_fraction=reach_fraction, remove_inverses=remove_inverses, create_summary=create_summary, separate_types=separate_types)
	
	# We split the graph
	splitter = None
	if(splitting_technique == "statistical"):
		splitter = StatistitalSplitter(pvalue_threshold)
	else:
		splitter = RandomSplitter(testing_fraction)
	splitter.setKG(kgd)
	kgd.graphs = splitter.split()
	kgd.graphs[0]['valid'] = dict()
	kgd.graphs[0]['valid']['positive'] = set()
	kgd.graphs[0]['valid']['negative'] = set()

	if(validation_fraction > 0):
		kgd.create_validation(validation_fraction)

	# We generate the negatives
	for generator in negatives_generators:
		generator.setKG(kgd)
		generator.initialize()
	kgd.generate_negatives(0, "test", negatives_generators, True, False)
	if(generate_negatives_training):
		kgd.generate_negatives(0, "train", negatives_generators, True, False)
	# We export the files
	kgd.export_files(0, True, include_data_prop, separate_types)
	if(export_gexf):
		kgd.export_gexf(0, True, True, True, True)

def main():
	parser = argparse.ArgumentParser(prog="AYNEC DataGen", fromfile_prefix_chars='@', description='Generates evaluation datasets from knowledge graphs.')
	parser.add_argument('--version', action='version', version='%(prog)s' + VERSION)
	parser.add_argument('--inF', required=True, help='The input file to read the original knowledge graph from')
	parser.add_argument('--outF', required=True, help='The folder where the output will be stored. If the folder does not exist, it will be created')
	parser.add_argument('--format', choices=['rdfa', 'xml', 'nt', 'n3', 'trix', 'simpleTriplesReader'], default='simpleTriplesReader', help='The format of the input file')
	parser.add_argument('--fractionAll', type=float, default=1.0, help='The overall fraction to take from the graph. The fraction is not the exact final fraction, but the probability of keeping each edge.')
	parser.add_argument('--minNumRel', type=int, default=2, help='Minimum frequency required to keep a relation during preprocessing')
	parser.add_argument('--reachFraction', type=float, default=1.0, help='Fraction of the total number of edges to keep during preprocessing, accumulating the relations, sorted by frequency. Use 1.0 to keep all edges')
	parser.add_argument('--removeInv', action='store_true', help='Specify if detected inverses should be removed during preprocessing')
	parser.add_argument('--thresInv', type=float, default=0.9, help='The overlap threshold used to detect inverses. For a pair to be detected as inverses, both relations must have a fraction of their edges as inverses in the other relation above the given threshold')
	parser.add_argument('--notCreateSum', action='store_false', help='Specify if you do not want to create an html summary of the relations frequency and the entities degree')
	parser.add_argument('--computePPR', action='store_true', help='Specify to compute the personalised page rank (PPR) of each node in the graph. So far this is only useful when generating negatives with the "PPR" strategy, so it should be set to False if it is not used')
	parser.add_argument('--fractionTest', type=float, default=0.2, help='Fraction of the edges used for testing')
	parser.add_argument('--fractionValidation', type=float, default=0.0, help='Fraction of the edges from the test sed to be used for validation')
	parser.add_argument('--splittingTechnique', choices=['random','statistical'], default='random', help='Algorithm employed to generate train/test splits out of the graph')
	parser.add_argument('--pValueThreshold', type=float, default=0.05, help='Threshold value for distribution comparation in statistical graph splitting technique')

	parser.add_argument('--change_target_kr', type=int, help='Generate the specified amount of negatives using the change target while keeping the range of the relations strategy')
	parser.add_argument('--change_source_kd', type=int, help='Generate the specified amount of negatives using the change source while keeping the domain of the relations strategy')
	parser.add_argument('--change_both_kdr', type=int, help='Generate the specified amount of negatives using the change both source and target while keeping the domain/range of the relations strategy')
	parser.add_argument('--change_target_random', type=int, help='Generate the specified amount of negatives using the change target at random strategy')
	parser.add_argument('--change_source_random', type=int, help='Generate the specified amount of negatives using the change source at random strategy')
	parser.add_argument('--change_both_random', type=int, help='Generate the specified amount of negatives using the change source at random strategy')
	parser.add_argument('--change_both_PPR', type=int, help='Generate the specified amount of negatives using the PPR strategy')
	
	parser.add_argument('--notNegTraining', action='store_false', help='Specify if negatives should not be generated for the training set. If False, they are only generated for the testing set')
	parser.add_argument('--notExportGEXF', action='store_false', help='Specify if the dataset should not be exported as a gexf file, useful for visualisation')
	parser.add_argument('--excludeDataProp', action='store_true', help='Specify if the dataset should not include a file with the data properties associated to entities')
	parser.add_argument('--separateTypes', action='store_true', help='Specify if triples containing entity types should be separated from the rest and included in a separate file with the types of each entity')

	args = parser.parse_args()

	INPUT_FILE = args.inF
	INPUT_FORMAT = args.format
	OUTPUT_FOLDER = args.outF
	GRAPH_FRACTION = args.fractionAll
	GENERATE_NEGATIVES_TRAINING = args.notNegTraining
	REMOVE_INVERSES = args.removeInv
	MIN_NUM_REL = args.minNumRel
	REACH_FRACTION = args.reachFraction
	TESTING_FRACTION = args.fractionTest
	VALIDATION_FRACTION = args.fractionValidation
	SPLITTING_TECHNIQUE = args.splittingTechnique
	PVALUE_THRESHOLD = args.pValueThreshold
	GENERATORS = {}
	if(args.change_target_kr is not None):
		GENERATORS[RandomGenerator(True, False, True)] = args.change_target_kr
	if(args.change_source_kd is not None):
		GENERATORS[RandomGenerator(True, True, False)] = args.change_source_kd
	if(args.change_both_kdr is not None):
		GENERATORS[RandomGenerator(True, True, True)] = args.change_both_kdr
	if(args.change_target_random is not None):
		GENERATORS[RandomGenerator(False, False, True)] = args.change_target_random
	if(args.change_source_random is not None):
		GENERATORS[RandomGenerator(False, True, False)] = args.change_source_random
	if(args.change_both_random is not None):
		GENERATORS[RandomGenerator(False, True, True)] = args.change_both_random
	if(args.change_both_PPR is not None):
		GENERATORS[PPRGenerator()] = args.change_both_PPR
	EXPORT_GEXF = args.notExportGEXF
	CREATE_SUMMARY = args.notCreateSum
	INVERSE_THRESHOLD = args.thresInv
	INCLUDE_DATA_PROP = not args.excludeDataProp
	SEPARATE_TYPES = args.separateTypes

	generate_datasets(	INPUT_FILE,
						INPUT_FORMAT,
						OUTPUT_FOLDER,
						GRAPH_FRACTION,
						GENERATE_NEGATIVES_TRAINING,
						REMOVE_INVERSES,
						MIN_NUM_REL,
						REACH_FRACTION,
						TESTING_FRACTION,
						VALIDATION_FRACTION,
						SPLITTING_TECHNIQUE,
						PVALUE_THRESHOLD,
						GENERATORS,
						EXPORT_GEXF,
						CREATE_SUMMARY,
						INVERSE_THRESHOLD,
						INCLUDE_DATA_PROP,
						SEPARATE_TYPES)

if __name__ == '__main__':
	main()
