from fractions import Fraction as F
# SGD, one step, no momentum. Doc order: negate if maximize, then add weight decay.
def sgd_correct(theta, g, lr, wd, maximize):
    d = -g if maximize else g
    d = d + wd * theta
    return theta - lr * d
def sgd_v1(theta, g, lr, wd, maximize):          # v1 bug: negation applied after weight decay
    d = g + wd * theta
    d = -d if maximize else d
    return theta - lr * d
th, g, lr, wd = F(1), F(0), F(1,10), F(1,10)
for name, step in [("correct", sgd_correct), ("v1", sgd_v1)]:
    a = step(th, g, lr, wd, True)                  # maximize with gradient g
    b = step(th, -g, lr, wd, False)                # minimize with gradient -g
    print(f"SGD {name}: maximize(g)={a}  minimize(-g)={b}  relation holds: {a==b}")

# BatchNorm running mean, momentum=None (cumulative average), first batch, running_mean starts at 0.
def bn_running_mean(batch, buggy):                 # batch: list of rows, each row has C channel values
    C = len(batch[0]); count = 0; rm = [F(0)]*C
    if not buggy: count += 1
    for c in range(C):
        if buggy: count += 1                       # v1 bug: counter incremented inside the channel loop
        bm = sum(row[c] for row in batch) / len(batch)
        rm[c] = rm[c] + (bm - rm[c]) / count
    return rm
batch = [[F(1), F(10)], [F(3), F(14)]]           # batch means: 2 and 12
for buggy in (False, True):
    whole = bn_running_mean(batch, buggy)
    split = [bn_running_mean([[r[c]] for r in batch], buggy)[0] for c in range(2)]
    print(f"BN {'v1' if buggy else 'correct'}: whole={whole} per-channel split={split} relation holds: {whole==split}")
