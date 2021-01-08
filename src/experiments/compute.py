import torch
import json
import ast
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.cuda()
roberta.eval() 
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

count = 0
c = 0
d = 0
m = {'not_entailment' : [0,1],'entailment': [2]}
for line in open('simile-entail.txt'):
        line = ast.literal_eval(line) #json.loads(line.strip())
        premise = line["premise"]
        hypothesis = line["hypothesis"]
        count = count +1
        label = m[line["label"]]
        tokens = roberta.encode(premise,hypothesis)
        tokens1 = roberta.encode(hypothesis)
        val1 = roberta.predict('mnli', tokens).argmax().tolist()
        val2 = roberta.predict('mnli', tokens1).argmax().tolist()
        if val1 in m['not_entailment'] and line["label"]=="not_entailment":
                c = c+1
        if val1 in m['entailment'] and line["label"]=="entailment":
                c = c+1
        if val2 in m['not_entailment'] and line["label"]=="not_entailment":
                d = d+1
        if val2 in m['entailment'] and line["label"]=="entailment":
                d = d+1

print("Normal",float(c)/float(count))
print("Hypothesis Only",float(d)/float(count))