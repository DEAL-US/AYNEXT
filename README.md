# AYNEC

This repository contains the DataGen and ResTest tools, which are implemented as python scripts. To run them, check the parameters at the start of the python file, and run it from console. The python files contains documentation about every parameter and function.

The following files with format examples are provided: "WN11.txt" and "mockup-results.txt", corresponding to the input of the DataGen and ResTest tools. In "WN11.txt", each line contains a triple in the following order: source, relation, target. In "mockup-results.txt", each line contains the source, relation, target, ground-truth (gt), type (**P**ositive, **C**hange **S**ource, **C**hange **T**arget, or **C**hange **B**oth) and a column with the result of each compared technique. Please, note that the file is expected to have the same header format, but with different techniques.

This software is licensed under the GPLv3 licence.

## Use cases

Though the way completion techniques work is always very similar, there are several paradigms when applying them that may affect how testing is done. Next, we describe how our tools would be applied to three different paradigms:

### <s,r,t> → score

This is the standard case: the techniques are fed individual triples and output a score. Use DataGen to create a dataset with the desired number of negatives, then apply the technique to every triple in the testing set. Finally, apply ResTest to the results in order to obtain metrics.

### <s,r,?> → t / <?,r,t> → s

The query approach takes a query as input (i.e, "where was John born?"), and outputs a ranking with a score for every possible candidate entity. There are two possibilities:

* Use DataGen to create a dataset *with no negatives* that is, with 0 negatives per positive, or use an existing dataset with negatives in the testing set but ignore them. While applying the techniques, use each existing combination among the positives of source/relation or relation/target as queries. The positives of each query are those in the testing set. The negatives, any possible triple that is not in the testing or training set. Include both positives and negatives in the results file. Finally, apply ResTest to the results in order to obtain metrics, which will include MAP and MRR.
* If applying the technique to every possible candidate is too time-consuming, Use DataGen to create a dataset *with several negatives per positive*, which will represent the negative candidates of the query. Then, apply the technique and ResTest.

### training → set of <s,r,t>

In this approach, a technique would not classify or give a score to triples, but would take the training set and output a set of new triples. Use DataGen to create a dataset *with no negatives* that is, with 0 negatives per positive, or use an existing dataset with negatives in the testing set but ignore them. Then, apply the techniques, and include in the results file both the true positives of the testing set (if the technique did not output some, those would be false negatives), and the output triples of the technique (if some were not in the testing set, they would be false positives). If you want to account for true negatives, include any triple that is not in the training set, nor the testing set, nor the output of the technique.

Finally, apply ResTest to the results in order to obtain metrics.

## DataGen

The DataGen tool takes as input a knowledge graph file as input and generates training/testing datasets for completion, as well as several auxiliary files. The following parameters can be used for configuration:

--inF: The input file to read the original knowledge graph from.\
--outF: The folder where the output will be stored. If the folder does not exist, it will be created.\
--format: The format of the input file. Choices: 'rdf', 'simpleTriplesReader'. Default = 'simpleTriplesReader'.\
--fractionAll: The overall fraction to take from the graph. The fraction is not the exact final fraction, but the probability of keeping each edge. Default = 1.0.\
--minNumRel: Minimum frequency required to keep a relation during preprocessing. Default = 2\
--reachFraction: Fraction of the total number of edges to keep during preprocessing, accumulating the relations, sorted by frequency. Use 1.0 to keep all edges. Default = 1.0.\
--removeInv: Specify if detected inverses should be removed during preprocessing.\
--thresInv: The overlap threshold used to detect inverses. For a pair to be detected as inverses, both relations must have a fraction of their edges as inverses in the other relation above the given threshold. Default = 0.9.\
--notCreateSum: Specify if you do not want to create an html summary of the relations frequency and the entities degree.\
--computePPR: Specify to compute the personalised page rank (PPR) of each node in the graph. So far this is only useful when generating negatives with the "PPR" strategy, so it should be set to False if it is not used.\
--fractionTest: Fraction of the edges used for testing. Default = 0.2.\
--splittingTechnique: Algorithm employed to generate train/test splits out of the graph. Choices: 'random','statistical'. Default = 'random'.\
--pValueThreshold: Threshold value for distribution comparation in statistical graph splitting technique. Default = 0.05.\
--change_target_kr: Generate the specified amount of negatives using the change target while keeping the range of the relations strategy.\
--change_source_kd: Generate the specified amount of negatives using the change source while keeping the domain of the relations strategy.\
--change_both_kdr: Generate the specified amount of negatives using the change both source and target while keeping the domain/range of the relations strategy.\
--change_target_random: Generate the specified amount of negatives using the change target at random strategy.\
--change_source_random: Generate the specified amount of negatives using the change source at random strategy.\
--change_both_random: Generate the specified amount of negatives using the change source at random strategy.\
--change_both_PPR: Generate the specified amount of negatives using the PPR strategy.\
--notNegTraining: Specify if negatives should not be generated for the training set. If False, they are only generated for the testing set.\
--notExportGEXF: Specify if the dataset should not be exported as a gexf file, useful for visualisation.\

The next section describe how the steps of the workflow can be customised.

### Use example

Let us suppose that we want to generate an evaluation dataset from the WN11 file, which contains a triple in each line. We want to use 20% of the dataset for testing, generating 2 negatives per positive by randomly replacing the source entity of the positive triples, removing relations with less than 10 instances, removing inverses with an overlapping threshold of 0.9. We would place the file in the same folder as DataGen.py and run the following command line:

```
python DataGen.py --inF ./WN11.txt --outF ./WN11-dataset --minNumRel 10 --removeInv --change_source_random 2
```

Let us now suppose that we want to generate an evaluation dataset from a rdf graph in a file named "wikidata.rdf". We want to use 50% of the dataset for testing, but without generating negatives. We want to remove relations with less than 30 instances and only keep relations that cover 90% of the graph. We also want to remove inverses want to remove inverses with an overlapping threshold of 0.95. We don't want to generate the .gexf file. Finally, since the original file is large, we only want to use aroung 75% of its triples. We would run the following command: 

```
python DataGen.py --inF ./wikidata.rdf --outF ./wikidata-dataset --format rdf --fractionAll 0.75 --minNumRel 30 --reachFraction 0.9 --removeInv --thresInv 0.95 --fractionTest 0.5 --notExportGEXF
```

### Preprocessing
The basic data reading is performed by the "read" function of classes that extends the Reader class.

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

Splitting is performed by the "split" function in classes that extend the Splitter class. It must store information in a graphs dictionary. Several splits are generated and stored in the first index of "graphs". The second index separates the training and testing sets. The third index separates the positive and negative examples. During splitting, only the positive examples set is filled in.

Let us suppose that we want to make a modification to RandomSplitter, so that the testing edges are taken in a completely random way, instead of in a per-relation basis. We would turn the following piece of code:

```python
for i in trange(self.number_splits):
	graphs[i] = dict()
	graphs[i]["train"] = dict()
	graphs[i]["test"] = dict()
	graphs[i]["train"]["positive"] = set()
	graphs[i]["test"]["positive"] = set()
	graphs[i]["train"]["negative"] = set()
	graphs[i]["test"]["negative"] = set()
	for rel in tqdm(self.kg.relations):
		edges = [(rel, s, t, 'P') for s, t in self.kg.grouped_edges[rel]]
		offset = floor(len(edges) / self.number_splits * i)
		fraction_test = self.fraction_test_relations.get(rel, 0.0)
		num_test = floor(len(edges) * fraction_test)
		ids_test = [(offset + x) % len(edges) for x in range(0, num_test)]
		ids_train = [(offset + x) % len(edges) for x in range(num_test, len(edges))]
		edges_test = [edges[id] for id in ids_test]
		edges_train = [edges[id] for id in ids_train]
		graphs[i]["test"]["positive"].update(edges_test)
		graphs[i]["train"]["positive"].update(edges_train)
```
 
 Into the following one:
 
```python
for i in trange(self.number_splits):
	graphs[i] = dict()
	graphs[i]["train"] = dict()
	graphs[i]["test"] = dict()
	graphs[i]["train"]["positive"] = set()
	graphs[i]["test"]["positive"] = set()
	graphs[i]["train"]["negative"] = set()
	graphs[i]["test"]["negative"] = set()
		edges = self.kg.edges
		offset = floor(len(edges) / self.number_splits * i)
		fraction_test = self.fraction_test_relations.get(rel, 0.0)
		num_test = floor(len(edges) * fraction_test)
		ids_test = [(offset + x) % len(edges) for x in range(0, num_test)]
		ids_train = [(offset + x) % len(edges) for x in range(num_test, len(edges))]
		edges_test = [edges[id] for id in ids_test]
		edges_train = [edges[id] for id in ids_train]
		graphs[i]["test"]["positive"].update(edges_test)
		graphs[i]["train"]["positive"].update(edges_train)
 ```
 
 ### Negatives generation
 
The generation of negative examples is performed by the "generate_negatives" function of classes that extend the NegativesGenerator class. These are used by the generate_negatives function of the KGDataset class, which iterates over every positive in every testing set and delegates the generation of a number of negatives to negatives generators that represent different strategies. Note that there is a filter of the positive examples used to generate negative examples:

```python
if(positive[0] not in self.ignored_rels_positives):
```

self.ignored_rels_positives is an initially empty list where we store relations that are ignored. We include a relation in this list when it is impossible to generate any negative from a positive of the relation. This happens, for example, when we want to generate negatives by changing the target of a triple to another entity while keeping the range of the relation, but all instances of the relation have the same target. Adding this check makes it possible to quickly discard such relations. This option, however, can be toggled.

The generation of negatives themselves takes place in generate_negatives functions of generator classes that take as arguments the positive and the number of negatives to generate, with additional parameters being provided during initialization. Once such generator is defined, it can be included as an additional strategy. For example, let us suppose that we want to implement a negatives generation strategy that merely replaces the source of the triple with a fictional entity named "foo", and the target with a fictional entity named "bar". We would define the following function:

```python
def generate_negatives_foobar(self, positive, number_negatives):
	rel = positive[0]
	negatives = [(rel, "foo", "bar") for i in range(number_negatives)]
	return negatives
 ```

## ResTest

The ResTest tool takes as input a file containing the gorund truth and the score given by each technique to each triple being tested. The results of each technique should each be in a different column, with a header corresponding to the name of the technique. The score of the technique can be either a binary score of a continuous probability. The following parameters, found at the beginning of the ResTest.py file, can be used for easy configuration of testing parameters:

RESULTS_FILE -- The input file containing the results of each technique. It should contain the following rows: triple source, triple relation, triple target, ground truth, and a column for each technique's results. Please, see the provided example, mockup_results.txt.
METRICS_OUTPUT_FILE -- The name of the file where the metrics will be stored.
PVALUES_OUTPUT_FILE -- The name of the file where the p-values will be stored.
THRESHOLDS -- A list with the positive/negative thresholds that will be used when computing metrics and p-values.
TARGET_QUERY -- Whether or not use the query <source, relation, ?> to compute ranking related metrics (MRR and MAP)
SOURCE_QUERY -- Whether or not use the query <?, relation, target> to compute ranking related metrics (MRR and MAP)
ALPHA -- The significance threshold for the rejection of a null hypothesis. Only used for console messages.

TARGET_QUERY would ideally be used when negative examples have been generated by changing the target of positive examples. The same applies to SOURCE_QUERY and generation by changing the source.

We compute the following metrics: precision, recall, accuracy, MAP, MRR, and WMR. Our referential metric is precision. Recall is also useful but to a lesser extend (since recall does not matter if the knowledge extracted is not, in almost all cases, correct). We have included MAP and MRR, since they enjoy some popularity, but there are some concerns regarding them:

MRR:
* It can only be computed when a completion tehcnique outputs continuous scores.
* The value depends on the size of the ranking, which in turn depends on the number of generated negatives per positive.
* It assumes that there is only one true positive result for each query, which is not correct in the context of KG completion. For example, there could be several positives in the testing set for query <?, born_in, Spain>. In other words, it should not be applied to many-to-many relations, one-to-many for target queries, and many-to-one for source queries.

MAP:
* It can only be computed when a completion technique outputs continuous scores.
* It takes as positives the first N entries of the ranking, where N is the number of true positives. However, in real uses of completion techniques, the positives are either the triples with a score above a given threshold, or the triple with the highest score. In the later case (which assumes, as MRR, that there is only a single true positive), the ranking used for MAP would only have a single element, and the metric would only measure what % of queries have the correct result at the very top, which rather than MAP, is the percent of Hits@1.
* It assumes that the order of the positives matters, as is the case in a search engine, where the top results are more visible. However, in real uses of completion techniques, the order of the positives does not have any effect. This objection does not apply if only a single true positive is expected.

Still, they are useful metrics, and their relevancy depends on how the completion techniques will be used in production.

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
