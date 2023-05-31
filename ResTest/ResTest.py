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

RESULTS_FILE = "ResTest/mockup-results.txt"

METRICS_OUTPUT_FILE = "ResTest/mockup-metrics.csv"

PVALUES_OUTPUT_FILE = "ResTest/mockup-pvalues.csv"

THRESHOLDS = [0.3]

HITS_AT = [1,5]

TARGET_QUERY = True

SOURCE_QUERY = True

ALPHA = 0.05

results = pd.read_table(RESULTS_FILE, '\t', header=0)
techniques = list(results.columns[5:])
metrics = dict()
rels = set(results["relation"])
lines_metrics = []
lines_metrics.append(("technique", "threshold", "relation", "metric", "value"))

for technique in techniques:
	metrics[technique] = dict()
	metrics[technique][-1] = dict()
	for rel in rels:
		metrics[technique][-1][rel] = dict()
		metrics[technique][-1][rel]["CS"] = dict()
		metrics[technique][-1][rel]["CT"] = dict()
		for n in HITS_AT:
			metrics[technique][-1][rel][f"hits_at_{n}_CS"] = 0
			metrics[technique][-1][rel][f"hits_at_{n}_CT"] = 0
	for threshold in THRESHOLDS:
		metrics[technique][threshold] = dict()
		for rel in rels:
			metrics[technique][threshold][rel] = dict()
			metrics[technique][threshold][rel]["TP"] = 0
			metrics[technique][threshold][rel]["FP"] = 0
			metrics[technique][threshold][rel]["TN"] = 0
			metrics[technique][threshold][rel]["FN"] = 0
			metrics[technique][threshold][rel]["CS"] = dict()
			metrics[technique][threshold][rel]["CT"] = dict()

RRs = {technique:{"CS":{rel: [] for rel in rels}, "CT":{rel: [] for rel in rels}}  for technique in techniques}
APs = {technique:{"CS":{rel: [] for rel in rels}, "CT":{rel: [] for rel in rels}} for technique in techniques}
hits = {technique:{n: 0 for n in HITS_AT}}

def update_rr_ap(grouped, n_type):
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
						# Checking if the positive is within the n first for the hits@n metrics
						for n in HITS_AT:
							if(i+1 <= n):
								metrics[technique][-1][rel][f"hits_at_{n}_{n_type}"] += 1
						# We keep track of the true positives so far to compute AP
						TP += 1
					# AP is computed from the first N results, where N is the number of true positives
					if(i < numPositives):
						ap += TP / (i + 1)
				RRs[technique][n_type][rel].append(rr)
				APs[technique][n_type][rel].append(ap / numPositives)

# MRR and MAP from target queries, grouping by source and relation
if(TARGET_QUERY):
	rows = results[results["type"].isin(("P", "CT"))]
	grouped = dict(tuple(rows.groupby(["source", "relation"])[["target", "gt"] + techniques]))
	update_rr_ap(grouped, "CT")

# MRR and MAP from source queries, grouping by target and relation
if(SOURCE_QUERY):
	rows = results[results["type"].isin(("P", "CS"))]
	grouped = dict(tuple(rows.groupby(["target", "relation"])[["source", "gt"] + techniques]))
	update_rr_ap(grouped, "CS")

for technique in techniques:
	for rel in rels:
		C_s = len(results[(results["type"]=="CS") & (results["relation"]==rel)])
		C_o = len(results[(results["type"]=="CT") & (results["relation"]==rel)])
		num_positives = len(results[(results["type"]=="P") & (results["relation"]==rel)])
		r_s = []
		r_o = []
		if(SOURCE_QUERY):
			r_s = np.array([1/rr for rr in RRs[technique]["CS"][rel]])
			mrr_cs = np.mean(RRs[technique]["CS"][rel])
			map_cs = np.mean(APs[technique]["CS"][rel])
			if(np.isnan(mrr_cs)):
				mrr_cs = None
			if(np.isnan(map_cs)):
				map_cs = None
			metrics[technique][-1][rel]["CS"]["MRR"] = mrr_cs
			metrics[technique][-1][rel]["CS"]["MAP"] = map_cs
			lines_metrics.append((technique, -1, rel, "MRR_CS", mrr_cs))
			lines_metrics.append((technique, -1, rel, "MAP_CS", map_cs))
			lines_metrics.append((technique, -1, rel, "MAP_CS", map_cs))
			for n in HITS_AT:
				hits_at_n = metrics[technique][-1][rel][f"hits_at_{n}_CS"] / num_positives
				metrics[technique][-1][rel][f"hits_at_{n}_CS"] = hits_at_n
				lines_metrics.append((technique, -1, rel, f"hits_at_{n}_CS", hits_at_n))
		if(TARGET_QUERY):
			r_o = np.array([1/rr for rr in RRs[technique]["CT"][rel]])
			mrr_ct = np.mean(RRs[technique]["CT"][rel])	
			map_ct = np.mean(APs[technique]["CT"][rel])
			if(np.isnan(mrr_ct)):
				mrr_ct = None
			if(np.isnan(map_ct)):
				map_cs = None	
			metrics[technique][-1][rel]["CT"]["MRR"] = mrr_ct
			metrics[technique][-1][rel]["CT"]["MAP"] = map_ct
			lines_metrics.append((technique, -1, rel, "MRR_CT", mrr_cs))
			lines_metrics.append((technique, -1, rel, "MAP_CT", mrr_cs))
			for n in HITS_AT:
				hits_at_n = metrics[technique][-1][rel][f"hits_at_{n}_CT"] / num_positives
				metrics[technique][-1][rel][f"hits_at_{n}_CT"] = hits_at_n
				lines_metrics.append((technique, -1, rel, f"hits_at_{n}_CT", hits_at_n))

		sum_r_s = np.sum(C_s*np.log(r_s)) if len(r_s==0) else 0
		sum_r_o = np.sum(C_o*np.log(r_o)) if len(r_o==0) else 0

		wmr = np.exp((sum_r_s+sum_r_o)/(len(r_s)*C_s+len(r_o)*C_o))

		metrics[technique][-1][rel]["WMR"] = wmr
		lines_metrics.append((technique, -1, rel, "WMR", wmr))

	if(SOURCE_QUERY):
		MRR_CSs = list(filter(None.__ne__, [metrics[technique][-1][rel]["CS"]["MRR"] for rel in rels]))
		MAP_CSs = list(filter(None.__ne__, [metrics[technique][-1][rel]["CS"]["MAP"] for rel in rels]))
		metrics[technique][-1]["MRR_CSs"] = MRR_CSs
		metrics[technique][-1]["MAP_CSs"] = MAP_CSs
		for n in HITS_AT:
			hits_at_n = list(filter(None.__ne__, [metrics[technique][-1][rel][f"hits_at_{n}_CS"] for rel in rels]))
			metrics[technique][-1][f"hits_at_{n}_CSs"] = hits_at_n

	if(TARGET_QUERY):
		MRR_CTs = list(filter(None.__ne__, [metrics[technique][-1][rel]["CT"]["MRR"] for rel in rels]))
		MAP_CTs = list(filter(None.__ne__, [metrics[technique][-1][rel]["CT"]["MAP"] for rel in rels]))
		metrics[technique][-1]["MRR_CTs"] = MRR_CSs
		metrics[technique][-1]["MAP_CTs"] = MAP_CSs
		for n in HITS_AT:
			hits_at_n = list(filter(None.__ne__, [metrics[technique][-1][rel][f"hits_at_{n}_CT"] for rel in rels]))
			metrics[technique][-1][f"hits_at_{n}_CTs"] = hits_at_n

	WMRs = list(filter(None.__ne__, [metrics[technique][-1][rel]["WMR"] for rel in rels]))
	metrics[technique][-1]["WMRs"] = WMRs

	# Micro and macro average for ranking metrics
	metrics[technique][-1]["macro-average"] = dict()
	metrics[technique][-1]["micro-average"] = dict()

	if(SOURCE_QUERY):
		all_rrs_cs = [metrics[technique][-1][rel]["CS"]["MRR"] for rel in rels]
		all_map_cs = [metrics[technique][-1][rel]["CS"]["MAP"] for rel in rels]
		micro_avg_rrs_cs = np.mean(np.hstack(all_rrs_cs))
		macro_avg_rrs_cs = np.mean([np.mean(rrs) for rrs in all_rrs_cs])
		micro_avg_map_cs = np.mean(np.hstack(all_map_cs))
		macro_avg_map_cs = np.mean([np.mean(map) for map in all_map_cs])
		metrics[technique][-1]["macro-average"]["MRR_CS"] = macro_avg_rrs_cs
		metrics[technique][-1]["micro-average"]["MRR_CS"] = micro_avg_rrs_cs
		metrics[technique][-1]["macro-average"]["MAP_CS"] = macro_avg_map_cs
		metrics[technique][-1]["micro-average"]["MAP_CS"] = micro_avg_map_cs
		lines_metrics.append((technique, -1, "macro-average", "MRR_CS", macro_avg_rrs_cs))
		lines_metrics.append((technique, -1, "micro-average", "MRR_CS", micro_avg_rrs_cs))
		lines_metrics.append((technique, -1, "macro-average", "MAP_CS", macro_avg_map_cs))
		lines_metrics.append((technique, -1, "micro-average", "MAP_CS", micro_avg_map_cs))
		for n in HITS_AT:
			all_hits_at_cs = [metrics[technique][-1][rel][f"hits_at_{n}_CS"] for rel in rels]
			macro_avg_hits_at_cs = np.mean(all_hits_at_cs)
			metrics[technique][-1]["macro-average"][f"hits_at_{n}_CS"] = macro_avg_hits_at_cs
			lines_metrics.append((technique, -1, "macro-average", f"hits_at_{n}_CS", macro_avg_hits_at_cs))

	if(TARGET_QUERY):
		all_rrs_ct = [metrics[technique][-1][rel]["CT"]["MRR"] for rel in rels]
		all_map_ct = [metrics[technique][-1][rel]["CT"]["MAP"] for rel in rels]
		micro_avg_rrs_ct = np.mean(np.hstack(all_rrs_ct))
		macro_avg_rrs_ct = np.mean([np.mean(rrs) for rrs in all_rrs_ct])
		micro_avg_map_ct = np.mean(np.hstack(all_map_ct))
		macro_avg_map_ct = np.mean([np.mean(map) for map in all_map_ct])
		metrics[technique][-1]["macro-average"]["MRR_CT"] = macro_avg_rrs_ct
		metrics[technique][-1]["micro-average"]["MRR_CT"] = micro_avg_rrs_ct
		metrics[technique][-1]["macro-average"]["MAP_CT"] = macro_avg_map_ct
		metrics[technique][-1]["micro-average"]["MAP_CT"] = micro_avg_map_ct
		lines_metrics.append((technique, -1, "macro-average", "MRR_CT", macro_avg_rrs_ct))
		lines_metrics.append((technique, -1, "micro-average", "MRR_CT", micro_avg_rrs_ct))
		lines_metrics.append((technique, -1, "macro-average", "MAP_CT", macro_avg_map_ct))
		lines_metrics.append((technique, -1, "micro-average", "MAP_CT", micro_avg_map_ct))
		for n in HITS_AT:
			all_hits_at_ct = [metrics[technique][-1][rel][f"hits_at_{n}_CT"] for rel in rels]
			macro_avg_hits_at_ct = np.mean(all_hits_at_ct)
			metrics[technique][-1]["macro-average"][f"hits_at_{n}_CT"] = macro_avg_hits_at_ct
			lines_metrics.append((technique, -1, "macro-average", f"hits_at_{n}_CT", macro_avg_hits_at_ct))

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
pvalues_metrics = ["precisions", "recalls", "f1s", "WMRs"]
if(SOURCE_QUERY):
	pvalues_metrics.extend(("MRR_CSs", "MAP_CSs"))
	for n in HITS_AT:
		pvalues_metrics.append(f"hits_at_{n}_CSs")
if(TARGET_QUERY):
	pvalues_metrics.extend(("MRR_CTs", "MAP_CTs"))
	for n in HITS_AT:
		pvalues_metrics.append(f"hits_at_{n}_CTs")
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
						print(f'* There were no samples for technique {t0[0]} and metric {metric}, skipping test')
					elif(len(t1[1]) == 0):
						print(f'* There were no samples for technique {t1[0]} and metric {metric}, skipping test')
					else:
						p_ks_2samp = ks_2samp(t0[1], t1[1]).pvalue
						print(f'* Two samples KS (unpaired) for threshold {threshold} metric {metric}, {t0[0]}-{t1[0]}: {p_ks_2samp}')
						if(p_ks_2samp < ALPHA):
							print("There are significant differences (unpaired)")
						else:
							print("There are NOT significant differences (unpaired)")
						if(len(t0[1]) == len(t1[1])):
							p_wilcoxon = 1.0
							try:
								p_wilcoxon = wilcoxon(t0[1], t1[1]).pvalue
							except ValueError:
								print("Value error when computing Wilcoxon test. Probably caused by exactly similar distributions.")
							print(f'* Two samples Wilcoxon (paired) for threshold {threshold} metric {metric}, {t0[0]}-{t1[0]}: {p_wilcoxon}')							
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
