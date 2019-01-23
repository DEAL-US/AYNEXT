# AYNEC
Tools from the AYNEC suite

This repository contains the DataGen and ResTest tools, which are implemented as python scripts. To run them, check the parameters at the start of the python file, and run it from console. The python files contains documentation about every parameter and function.

The following files with format examples are provided: "WN11.txt" and "mockup-results.txt", corresponding to the input of the DataGen and ResTest tools. In "WN11.txt", each line contains a triple in the following order: source, relation, target. In "mockup-results.txt", each line contains the source, relation, target, ground-truth (gt), and a column with the result of each compared technique. Please, note that the file is expected to have the same header, but with different techniques.

This software is licensed under the GPLv3 licence. It is presented in the article "AYNEC: All You Need for Evaluating Completion Techniques in Knowledge Graphs", sent for the ESWC19 conference and currently under revision.

## DataGen

The DataGen tool takes as input a knowledge graph a file with a triple with file, with tabs as separators ("<source>  <relation>  <target>" for each line). The following parameters, found at the beginning of the DataGen.py file, can be used for easy configuration of dataset generation parameters and strategies:

INPUT_FILE -- The path of the input file to read the original knowledge graph from
OUTPUT_FOLDER -- The path of the folder where the output will be stored. If the folder does not exist, it will be created
GRAPH_FRACTION -- The overall fraction to take from the graph. The fraction is not the exact fraction, but the probability of keeping each edge.
GENERATE_NEGATIVES_TRAINING -- Whether or not negatives should be generated for the training set. If False, they are only generated for the testing set
REMOVE_INVERSES -- Whether or not detected inverses should be removed during preprocessing
MIN_NUM_REL -- Minimum frequency required to keep a relation during preprocessing
REACH_FRACTION -- Fraction of the total number of edges to keep during preprocessing, accumulating the relations, sorted by frequency. Use 1.0 to keep all edges
TESTING_FRACTION  -- Fraction used for testing
NUMBER_NEGATIVES -- Number of negatives to generate per positive
NEGATIVES_STRATEGY -- Strategy used to generate negatives. Possible: change_target, change_source, change_both_random, change_target_random, change_source_random, change_both_random, PPR
EXPORT_GEXF -- Whether or not the dataset should be exported as a gexf file, useful for visualisation
CREATE_SUMMARY -- Whether or not to create an html summary of the relations' frequency and the entities' degree
COMPUTE_PPR -- Whether or not to compute the personalised page rank (PPR) of each node in the graph. So far this is only useful when generating negatives with the "PPR" strategy, so it should be set to False if it is not used

The next section describe how the steps of the workflow can be customised.

### Preprocessing
The basic data reading is performed by the "read" function of a class that extends the Reader class. The provided implementation reads a file with a triple in each line.

A new implementation could be, for example, a reader that reads form a file with literals attached to each entity. In this case, the dictionary attached to each entity would include other keys apart from those corresponding to the degrees.

The preprocessing itself, however, is performed in the read function of the DatasetsGenerator class, which has most parameters relevant to preprocessing.

This class makes use of a Reader to obtain the dictionary/set of entities, relations and edges, and stores them in the attributes of the class. Any additional preprocessing would be included here. For example, let's assume we want to remove relations with name longer than 10 characters. We would do so by adding the following code before the "self.find_inverses()" line:

```python
removed_rels = [rel for rel in self.relations if len(rel) > 10]
print("\nRemoving relations with more than 10 characters")
self.remove_rels(removed_rels)
```

Note the call to "self.create_summary(accepted_rels, amounts, accumulated_fractions)", which creates the html summary from the provided relations and their frequency. We generate it before removing inverses (if specified so) in order to not to mess with the displayed accumulated frequency. It is also in this function that we store any information about the graph in any inner attribute that is considered necessary. For example, the following piece of code stores the outgoing edge of each entity node:

```python
print("Storing outgoing edges for each node")
for edge in tqdm(self.edges):
  if(edge[1] not in self.entity_edges):
    self.entity_edges[edge[1]] = list()
  self.entity_edges[edge[1]].append(edge)
```

### Splitting
Splitting is performed in the split_graph function. It stores information int the "graphs" dictionary. Several splits are generated and stored in the first index of "graphs". The second index separates the training and testing sets. The third index separates the positive and negative examples. During splitting, only the positive examples set is filled in. For the edges of each relation, a set of ids for training and testing set is used to split the graph.

Let us suppose that we want to make a modification, so that the testing edges are taken in a completely random way, instead of in a per-relation basis. We would turn the following piece of code:

```python
for i in trange(self.number_splits):
	self.graphs[i] = dict()
	self.graphs[i]["train"] = dict()
	self.graphs[i]["test"] = dict()
	self.graphs[i]["train"]["positive"] = set()
	self.graphs[i]["test"]["positive"] = set()
	for rel in tqdm(self.relations):
		edges = [(rel, s, t) for s, t in self.grouped_edges[rel]]
		offset = floor(len(edges) / self.number_splits * i)
		fraction_test = fraction_test_relations.get(rel, 0.0)
		num_test = floor(len(edges) * fraction_test)
		ids_test = [(offset + x) % len(edges) for x in range(0, num_test)]
		ids_train = [(offset + x) % len(edges) for x in range(num_test, len(edges))]
		edges_test = [edges[id] for id in ids_test]
		edges_train = [edges[id] for id in ids_train]
		self.graphs[i]["test"]["positive"].update(edges_test)
		self.graphs[i]["train"]["positive"].update(edges_train)
	```
 
 Into the following one:
 
 ```python
for i in trange(self.number_splits):
	self.graphs[i] = dict()
	self.graphs[i]["train"] = dict()
	self.graphs[i]["test"] = dict()
	self.graphs[i]["train"]["positive"] = set()
	self.graphs[i]["test"]["positive"] = set()
	offset = floor(len(self.edges) / self.number_splits * i)
	num_test = floor(len(self.edges) * fraction_test)
	ids_test = [(offset + x) % len(self.edges) for x in range(0, num_test)]
	ids_train = [(offset + x) % len(self.edges) for x in range(num_test, len(self.edges))]
	edges_test = [self.edges[id] for id in ids_test]
	edges_train = [self.edges[id] for id in ids_train]
	self.graphs[i]["test"]["positive"].update(edges_test)
	self.graphs[i]["train"]["positive"].update(edges_train)
 ```
 
 ### Negatives generation
 
The generation of negative examples is performed in function generate_negatives. This function iterates over every positive in every testing set and delegates the generation of a number of negatives to functions that represent different strategies. Note that there is a filter of the positive examples used to generate negative examples:

```python
if(positive[0] not in self.ignored_rels_positives):
```

self.ignored_rels_positives is an initially empty list where we store relations that are ignored. We include a relation in this list when it is impossible to generate any negative from a positive of the relation. This happens, for example, when we want to generate negatives by changing the target of a triple to another entity while keeping the range of the relation, but all instances of the relation have the same target. Adding this check makes it possible to quickly discard such relations. This option, however, can be toggled.

The generation of negatives themselves takes place in functions that take as arguments, at the very least, the positive and the number of negatives to generate. Once such function is defined, it can be included as an additional strategy. For example, let us suppose that we want to implement a negatives generation strategy that merely replaces the source of the triple with a fictional entity named "foo", and the target with a fictional entity named "bar". We would define the following function:

```python
def generate_negatives_foobar(self, positive, number_negatives):
	rel = positive[0]
	negatives = [(rel, "foo", "bar") for i in range(number_negatives)]
	return negatives
 ```
 
 And it could be included with other strategies:
 
 ```python
	...
	if(strategy == "change_source"):
		new_negatives = self.generate_negatives_random(positive, num_negatives, True, True, False)
	elif(strategy == "change_target"):
		new_negatives = self.generate_negatives_random(positive, num_negatives, True, False, True)
	elif(strategy == "change_both"):
		new_negatives = self.generate_negatives_random(positive, num_negatives, True, True, True)
	elif(strategy == "change_source_random"):
		new_negatives = self.generate_negatives_random(positive, num_negatives, False, True, False)
	elif(strategy == "change_target_random"):
		new_negatives = self.generate_negatives_random(positive, num_negatives, False, False, True)
	elif(strategy == "change_both_random"):
		new_negatives = self.generate_negatives_random(positive, num_negatives, False, True, True)
	elif(strategy == "PPR"):
		new_negatives = self.generate_negatives_PPR(positive, num_negatives)
	elif(strategy == "foobar"):
		new_negatives = self.generate_negatives_foobar(positive, num_negatives)
```
