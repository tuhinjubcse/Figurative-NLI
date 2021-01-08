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
m = {0: 'not_entailment' , 1: 'not_entailment', 2:'entailment'}
for line in open('simile-entail.txt'):
        line = ast.literal_eval(line) #json.loads(line.strip())
        premise = line["premise"]
        hypothesis = line["hypothesis"]
        count = count +1
        tokens = roberta.encode(premise,hypothesis)
        tokens1 = roberta.encode(hypothesis)
        val1 = roberta.predict('mnli', tokens).argmax().tolist()
        val2 = roberta.predict('mnli', tokens1).argmax().tolist()
        if m[val1]==line["label"]:
                c = c+1
        else:
                print(line)
        if m[val2]==line["label"]:
                d = d+1

print("Normal",float(c)/float(count))
print("Hypothesis Only",float(d)/float(count))