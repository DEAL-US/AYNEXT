import pandas as pd
from scipy.stats import ks_2samp, wilcoxon
from itertools import combinations
import numpy as np

"""
This script computes metrics from triple classification results from several techniques.

The following arguments can be used for simple configuration
RESULTS_FILE -- The input file containing the results of each technique. It should contain the following rows: triple source, triple relation, triple target, ground truth, and a column for each technique's results. Please, see the provided example, mockup_results.txt.
METRICS_OUTPUT_FILE -- The name of the file where the metrics will be stored.
PVALUES_OUTPUT_FILE -- The name of the file where the p-values will be stored.
THRESHOLDS -- A list with the positive/negative thresholds that will be used when computing metrics and p-values.
TARGET_QUERY -- Whether or not use the query <source, relation, ?> to compute ranking related metrics (MRR and MAP)
SOURCE_QUERY -- Whether or not use the query <?, relation, target> to compute ranking related metrics (MRR and MAP)
ALPHA -- The significance threshold for the rejection of a null hypothesis. Only used for console messages.
"""

RESULTS_FILE = "./mockup-results.txt"

METRICS_OUTPUT_FILE = "./mockup-metrics.csv"

PVALUES_OUTPUT_FILE = "./mockup-pvalues.csv"

THRESHOLDS = [0.3]

TARGET_QUERY = True

SOURCE_QUERY = True

ALPHA = 0.05

results = pd.read_table(RESULTS_FILE, '\t', header=0)
techniques = list(results.columns[4:])
metrics = dict()
rels = set(results["relation"])
lines_metrics = []
lines_metrics.append(("technique", "threshold", "relation", "metric", "value"))

for technique in techniques:
	metrics[technique] = dict()
	metrics[technique][-1] = dict()
	for rel in rels:
		metrics[technique][-1][rel] = dict()
	for threshold in THRESHOLDS:
		metrics[technique][threshold] = dict()
		for rel in rels:
			metrics[technique][threshold][rel] = dict()
			metrics[technique][threshold][rel]["TP"] = 0
			metrics[technique][threshold][rel]["FP"] = 0
			metrics[technique][threshold][rel]["TN"] = 0
			metrics[technique][threshold][rel]["FN"] = 0

RRs = {technique: {rel: [] for rel in rels} for technique in techniques}
APs = {technique: {rel: [] for rel in rels} for technique in techniques}

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

# MRR and MAP from target queries, grouping by source and relation
if(TARGET_QUERY):
	grouped = dict(tuple(results.groupby(["source", "relation"])[["target", "gt"] + techniques]))
	update_rr_ap(grouped)

# MRR and MAP from source queries, grouping by target and relation
if(SOURCE_QUERY):
	grouped = dict(tuple(results.groupby(["target", "relation"])[["source", "gt"] + techniques]))
	update_rr_ap(grouped)

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

# Computing the confusion matrix
for index, row in results.iterrows():
	gt = row["gt"]
	rel = row["relation"]
	for technique in techniques:
		result = row[technique]
		for threshold in THRESHOLDS:
			if(gt == 1):
				if(result >= threshold):
					metrics[technique][threshold][rel]["TP"] += 1
				else:
					metrics[technique][threshold][rel]["FN"] += 1
			else:
				if(result >= threshold):
					metrics[technique][threshold][rel]["FP"] += 1
				else:
					metrics[technique][threshold][rel]["TN"] += 1

def computeF(precision, recall, f):
	if (precision is None or recall is None):
		res = None
	elif((precision + recall) == 0):
		res = 0.0
	else:
		res = (1 + f * f) * precision * recall / ((f * f * precision) + recall)
	return res

# Computing the set metrics
for technique in techniques:
	print(technique)
	for threshold in THRESHOLDS:
		metrics[technique][threshold]["macro-average"] = dict()
		metrics[technique][threshold]["micro-average"] = dict()
		print(f'\t{threshold}')

		# Precision, recall, and f1 for each relation
		for rel in rels:
			print(f'\t\t{rel}')
			all_predicted_positives = (metrics[technique][threshold][rel]["TP"] + metrics[technique][threshold][rel]["FP"])
			if (all_predicted_positives == 0):
				metrics[technique][threshold][rel]["precision"] = None
			else:
				metrics[technique][threshold][rel]["precision"] = metrics[technique][threshold][rel]["TP"] / all_predicted_positives

			all_true_positives = (metrics[technique][threshold][rel]["TP"] + metrics[technique][threshold][rel]["FN"])
			if (all_true_positives == 0):
				metrics[technique][threshold][rel]["recall"] = None
			else:
				metrics[technique][threshold][rel]["recall"] = metrics[technique][threshold][rel]["TP"] / (metrics[technique][threshold][rel]["TP"] + metrics[technique][threshold][rel]["FN"])

			metrics[technique][threshold][rel]["f1"] = computeF(metrics[technique][threshold][rel]["precision"], metrics[technique][threshold][rel]["recall"], 1)

			print(f'\t\t\tPrecision: {metrics[technique][threshold][rel]["precision"]}')
			print(f'\t\t\tRecall: {metrics[technique][threshold][rel]["recall"]}')
			print(f'\t\t\tF1: {metrics[technique][threshold][rel]["f1"]}')
			lines_metrics.append((technique, threshold, rel, "precision", metrics[technique][threshold][rel]["precision"]))
			lines_metrics.append((technique, threshold, rel, "recall", metrics[technique][threshold][rel]["recall"]))
			lines_metrics.append((technique, threshold, rel, "f1", metrics[technique][threshold][rel]["f1"]))

		# Macro precision, recall, and f1
		precisions = list(filter(None.__ne__, [metrics[technique][threshold][rel]["precision"] for rel in rels]))
		metrics[technique][threshold]["precisions"] = precisions
		if (len(precisions) == 0):
			metrics[technique][threshold]["macro-average"]["precision"] = None
		else:
			metrics[technique][threshold]["macro-average"]["precision"] = sum(precisions) / len(precisions)

		recalls = list(filter(None.__ne__, [metrics[technique][threshold][rel]["recall"] for rel in rels]))
		metrics[technique][threshold]["recalls"] = recalls
		if (len(recalls) == 0):
			metrics[technique][threshold]["macro-average"]["recall"] = None
		else:
			metrics[technique][threshold]["macro-average"]["recall"] = sum(recalls) / len(recalls)

		f1s = list(filter(None.__ne__, [metrics[technique][threshold][rel]["f1"] for rel in rels]))
		metrics[technique][threshold]["f1s"] = f1s
		if (len(f1s) == 0):
			metrics[technique][threshold]["macro-average"]["f1"] = None
		else:
			metrics[technique][threshold]["macro-average"]["f1"] = sum(f1s) / len(f1s)

		print(f'\t\tMacro precision: {metrics[technique][threshold]["macro-average"]["precision"]}')
		print(f'\t\tMacro recall: {metrics[technique][threshold]["macro-average"]["recall"]}')
		print(f'\t\tMacro f1: {metrics[technique][threshold]["macro-average"]["f1"]}')

		lines_metrics.append((technique, threshold, "macro-average", "precision", metrics[technique][threshold]["macro-average"]["precision"]))
		lines_metrics.append((technique, threshold, "macro-average", "recall", metrics[technique][threshold]["macro-average"]["recall"]))
		lines_metrics.append((technique, threshold, "macro-average", "f1", metrics[technique][threshold]["macro-average"]["f1"]))

		# Micro precision, recall, and f1
		total_TP = sum([metrics[technique][threshold][rel]["TP"] for rel in rels])
		total_FP = sum([metrics[technique][threshold][rel]["FP"] for rel in rels])
		total_TN = sum([metrics[technique][threshold][rel]["TN"] for rel in rels])
		total_FN = sum([metrics[technique][threshold][rel]["FN"] for rel in rels])

		if (total_TP + total_FP == 0):
			metrics[technique][threshold]["micro-average"]["precision"] = None
		else:
			metrics[technique][threshold]["micro-average"]["precision"] = total_TP / (total_TP + total_FP)
		if (total_TP + total_FN == 0):
			metrics[technique][threshold]["micro-average"]["recall"] = None
		else:
			metrics[technique][threshold]["micro-average"]["recall"] = total_TP / (total_TP + total_FN)
		metrics[technique][threshold]["micro-average"]["f1"] = computeF(metrics[technique][threshold]["micro-average"]["precision"], metrics[technique][threshold]["micro-average"]["recall"], 1)

		print(f'\t\tMicro precision: {metrics[technique][threshold]["micro-average"]["precision"]}')
		print(f'\t\tMicro recall: {metrics[technique][threshold]["micro-average"]["recall"]}')
		print(f'\t\tMicro f1: {metrics[technique][threshold]["micro-average"]["f1"]}')

		lines_metrics.append((technique, threshold, "micro-average", "precision", metrics[technique][threshold]["micro-average"]["precision"]))
		lines_metrics.append((technique, threshold, "micro-average", "recall", metrics[technique][threshold]["micro-average"]["recall"]))
		lines_metrics.append((technique, threshold, "micro-average", "f1", metrics[technique][threshold]["micro-average"]["f1"]))

pvalues = dict()
pvalues_metrics = ["precisions", "recalls", "f1s", "MRRs", "MAPs"]
THRESHOLDS.append(-1)

# Computing the p-values
for threshold in THRESHOLDS:
	pvalues[threshold] = dict()
	for metric in pvalues_metrics:
		if metric in metrics[technique][threshold]:
			measures = [(technique, metrics[technique][threshold][metric]) for technique in techniques]
			pvalues[threshold][metric] = dict()
			if(len(measures) >= 2):
				for combination in combinations(measures, 2):
					t0 = combination[0]
					t1 = combination[1]
					p_ks_2samp = None
					p_wilcoxon = None
					pvalues[threshold][metric][(t0[0], t1[0])] = dict()
					if(len(t0[1]) == 0):
						print(f'* There were no samples for technique {t0[0]}, skipping test')
					elif(len(t1[1]) == 0):
						print(f'* There were no samples for technique {t1[0]}, skipping test')
					else:
						p_ks_2samp = ks_2samp(t0[1], t1[1]).pvalue
						print(f'* Two samples KS (unpaired) for threshold {threshold}, {t0[0]}-{t1[0]}: {p_ks_2samp}')
						if(p_ks_2samp < ALPHA):
							print("There are significant differences (unpaired)")
						else:
							print("There are NOT significant differences (unpaired")
						if(len(t0[1]) == len(t1[1])):
							p_wilcoxon = wilcoxon(t0[1], t1[1]).pvalue
							print(f'* Two samples Wilcoxon (paired) for threshold {threshold}, {t0[0]}-{t1[0]}: {p_wilcoxon}')
							if(p_wilcoxon < ALPHA):
								print("There are significant differences (paired)")
					pvalues[threshold][metric][(t0[0], t1[0])]["KS"] = p_ks_2samp
					pvalues[threshold][metric][(t0[0], t1[0])]["Wilcoxon"] = p_wilcoxon

# Storing the lines to be written regarding p-values
lines_pvalues = []
lines_pvalues.append(("tech1", "tech2", "threshold", "metric", "test", "p-value"))
for threshold in THRESHOLDS:
	for metric in pvalues_metrics:
		if(metric in pvalues[threshold]):
			combinations = pvalues[threshold][metric].keys()
			for combination in combinations:
				tests = pvalues[threshold][metric][combination]
				for test in tests:
					lines_pvalues.append((combination[0], combination[1], threshold, metric, test, pvalues[threshold][metric][combination][test]))

with open(METRICS_OUTPUT_FILE, "w") as file:
	content = ["\t".join(str(element) for element in line) for line in lines_metrics]
	file.writelines("\n".join(content))

with open(PVALUES_OUTPUT_FILE, "w") as file:
	content = ["\t".join(str(element) for element in line) for line in lines_pvalues]
	file.writelines("\n".join(content))
