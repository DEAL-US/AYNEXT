# AYNEC
Tools from the AYNEC suite

This repository contains the DataGen and ResTest tools, which are implemented as python scripts. To run them, check the parameters at the start of the python file, and run it from console. The python files contains documentation about every parameter and function.

The following files with format examples are provided: "WN11.txt" and "mockup-results.txt", corresponding to the input of the DataGen and ResTest tools. In "WN11.txt", each line contains a triple in the following order: source, relation, target. In "mockup-results.txt", each line contains the source, relation, target, ground-truth (gt), and a column with the result of each compared technique. Please, note that the file is expected to have the same header, but with different techniques.

This software is licensed under the GPLv3 licence. It is presented in the article "AYNEC: All You Need for Evaluating Completion Techniques in Knowledge Graphs", sent for the ESWC19 conference and currently under revision.

## Use cases

Though the way completion techniques work is always very similar, there are several paradigms when applying them that may affect how testing is done. Next, we describe how our tools would be applied to three different paradigms:

### <s,r,t> → score

This is the standard case: the techniques are fed individual triples and output a score. Use DataGen to create a dataset with the desired number of negatives, then apply the technique to every triple in the testing set. Finally, apply ResTest to the results in order to obtain metrics.

### <s,r,?> → t / <?,r,t> → s

The query approach takes a query as input (i.e, "where was John born?"), and outputs a ranking with a score for every possible candidate entity. There are two possibilities:

* Use DataGen to create a dataset *with no negatives* that is, with 0 negatives per positive, or use an existing dataset with negatives in the testing set but ignore them. While applying the techniques, use each existing combination among the positives of source/relation or relation/target as queries. The positives of each query are those in the testing set. The negatives, any possible triple that is not in the testing or training set. Include both positives and negatives in the results file. Finally, apply ResTest to the results in order to obtain metrics, which will include MAP and MRR.
* If applying the technique to every possible candidate is too time-consuming, Use DataGen to create a dataset *with several negatives per positive*, which will represent the negative candidates of the query. Then, apply the technique and ResTest.

### training → set of <s,r,t>

In this approach, a technique would not classify or give a score to triples, but would take the training set and output a set of new triples. Use DataGen to create a dataset *with no negatives* that is, with 0 negatives per positive, or use an existing dataset with negatives in the testing set but ignore them. Then, apply the techniques, and include in the results file both the true positives of the testing set (if the technique did not output some, those would be false negatives), and the output triples of the technique (if some were not in the testing set, they would be false positives). If you want to account for true positives, include any triple that is not in the training set, nor the testing set, nor the output of the technique.

Finally, apply ResTest to the results in order to obtain metrics.

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
INVERSE_THRESHOLD -- The overlap threshold used to detect inverses. For a pair to be detected as inverses, both relations must have a fraction of their edges as inverses in the other relation above the given threshold.

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

## ResTest

The ResTest tool takes as input a file containing the gorund truth and the score given by each technique to each triple being tested. The results of each technique should each be ina  different column, with a header corresponding to the name of the technique. The score of the technique can be either a binary score of a continuous probability. The following parameters, found at the beginning of the ResTest.py file, can be used for easy configuration of testing parameters:

RESULTS_FILE -- The input file containing the results of each technique. It should contain the following rows: triple source, triple relation, triple target, ground truth, and a column for each technique's results. Please, see the provided example, mockup_results.txt.
METRICS_OUTPUT_FILE -- The name of the file where the metrics will be stored.
PVALUES_OUTPUT_FILE -- The name of the file where the p-values will be stored.
THRESHOLDS -- A list with the positive/negative thresholds that will be used when computing metrics and p-values.
TARGET_QUERY -- Whether or not use the query <source, relation, ?> to compute ranking related metrics (MRR and MAP)
SOURCE_QUERY -- Whether or not use the query <?, relation, target> to compute ranking related metrics (MRR and MAP)
ALPHA -- The significance threshold for the rejection of a null hypothesis. Only used for console messages.

TARGET_QUERY would ideally be used when negative examples have been generated by changing the target of positive examples. The same applies to SOURCE_QUERY and generation by changing the source.

We compute the following metrics: precision, recall, accuracy, MAP, and MRR. Our referential metric is precision. Recall is also useful but to a lesser extend (since recall does not matter if the knowledge extracted is not, in almost all cases, correct). We have included MAP and MRR, since they enjoy some popularity, but there are some concerns regarding them:

MRR:
* It can only be computed when a completion tehcnique outputs continuous scores.
* The value depends on the size of the ranking, which in turn depends on the number of generated negatives per positive.
* It assumes that there is only one true positive result for each query, which is not correct in the context of KG completion. For example, there could be several positives in the testing set for query <?, born_in, Spain>.

MAP:
* It can only be computed when a completion technique outputs continuous scores.
* It takes as positives the first N entries of the ranking, where N is the number of true positives. However, in real uses of completion techniques, the positives are either the triples with a score above a given threshold, or the triple with the highest score. In the later case (which assumes, as MRR, that there is only a single true postiive), the ranking used for MAP would only have a single element, and the metric would only measure what % of queries have the correct result at the very top, which rather than MAP, is a kind of "query accuracy".
* It assumes that the order of the positives matter, as is the case in a search engine, where the top results are more visible. However, in real uses of completion techniques, the order of the positives does not have any effect. This objection does not apply if onle a single true positive is expected.

Still, they are useful metric, and their relevancy depends on how the completion techniques will be used in production.

All metrics are stored in a dictionary called metrics, which has the following structure:

metrics\[technique]\[threshold]\[relation]\[metric name] = x

Where threshold can be set to -1 if the metric does not depend on any threshold

### Ranking based metrics

The basis of ranking based metrics are queries as the following one: "what is John's father?", that is, what is the source of triple <?, father_of, John>. The former query is a source query, since it tries to infer the missing source in a triple. A target query would be, for example, "where was John born?": <John, born_in, ?>

Each query has as potential results all entities in the graph, which would be sorted by score in order to then select either the top result (which assumes there is only one target/source), or those above a threshold.

We can simulate the testing of these queries by grouping the testing triples by source/relation, or by relation/target. This is done with the following code, in the case of target queries:

```python
grouped = dict(tuple(results.groupby(["source", "relation"])[["target", "gt"] + techniques]))
```

We can then go over each group, which represents the ranking of a query. For each group, and for each technique, we can sort the entries by the score given by the technique, and easily compute any metric while having access to the score, the position in the ranking, and whether or not it was a true postive (the ground truth):

```python
def update_rr_ap(grouped):
	for key, value in grouped.items():
		positives = value[value["gt"] == 1]
		numPositives = len(positives)
		if(numPositives > 0):
			rel = key[1]
			for technique in techniques:
				rr = 0
				TP = 0
				ap = 0
				for i, row in enumerate(value.sort_values([technique], ascending=False).iterrows()):
					score = row[1][technique]
					gt = row[1]["gt"]
					if(gt == 1):
						# RR is computed using only the position of the first positive
						if(rr == 0):
							rr = 1 / (i + 1)
						# We keep track of the true positives so far to compute AP
						TP += 1
					# AP is computed from the first N results, where N is the number of true positives
					if(i < numPositives):
						ap += TP / (i + 1)
				RRs[technique][rel].append(rr)
				APs[technique][rel].append(ap / numPositives)
```

It is in this loop that new ranking-based metrics would be implemented. The finally computed metrics are stored later:
```python
for technique in techniques:
	for rel in rels:
		mrr = np.mean(RRs[technique][rel])
		MAP = np.mean(APs[technique][rel])
		if(np.isnan(mrr)):
			mrr = None
		if(np.isnan(MAP)):
			MAP = None		
		metrics[technique][-1][rel]["MRR"] = mrr
		metrics[technique][-1][rel]["MAP"] = MAP
		lines_metrics.append((technique, -1, rel, "MRR", mrr))
		lines_metrics.append((technique, -1, rel, "MAP", MAP))

	MRRs = list(filter(None.__ne__, [metrics[technique][-1][rel]["MRR"] for rel in rels]))
	MAPs = list(filter(None.__ne__, [metrics[technique][-1][rel]["MAP"] for rel in rels]))

	metrics[technique][-1]["MRRs"] = MRRs
	metrics[technique][-1]["MAPs"] = MAPs
```

The last lines store the value of the metrics for each relation in a single array, which is used when computing p-values.

### Set based metrics

Set based metrics are computed from the confusion matrix of each technique/relation/threshold. ResTest first computes the confusion matrix in a loop, and the uses a different loop to compute the metrics. It is in that loop that new metrics could be included:

```python
for technique in techniques:
	print(technique)
	for threshold in THRESHOLDS:
		metrics[technique][threshold]["macro-average"] = dict()
		metrics[technique][threshold]["micro-average"] = dict()
		print(f'\t{threshold}')

		# Precision, recall, and f1 for each relation
		for rel in rels:
			# New metrics would be computed here, and stored in the relevant variables
		#Or here, if the metrics are not computed in a per-relation basis
```

## Future work

* <s>Add an option to consider two relation inverses when they overlap to a certain degree, instead of only when there is perfect overlapping.</s>
* Add additional export formats for the training and test sets
