import json

tp = 1871 + 196 + 3 + 4 + 28 + 27 + 4 + 170 + 314
tn = 1039.2 + 10067.3 + 749.9 + 749.9 + 150.7 + 161.7 + 14804 + 17119.4 + 19627.4
fp = 1 + 0.3 + 0.4 + 0.9 + 0.5 + 28.5 + 29.3 + 28.0
fn = 0

acc = (tp + tn)/(tp + tn + fp + fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
 
f2 = (5*precision*recall)/((4*precision)+recall)

with open("sofia.json", 'w') as f:
    json.dump({
        "acc":acc,
        "f2": f2
    }, f)