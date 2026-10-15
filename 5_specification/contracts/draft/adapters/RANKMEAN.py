from common import *
import spec_training_program as tp
def reference(inp):
    try: return bounds(tp.cross_rank_mean([q(s) for s in inp['S']], [int(n) for n in inp['n']]))
    except Exception as e:
        return UNDEF if 'no valid tokens' in str(e) else 'not established'
def M1(inp):
    means = [mpq(s) / n for s, n in zip(inp['S'], inp['n']) if n > 0]; return UNDEF if not means else mp.fsum(means) / len(means)
def M2(inp): return mp.fsum(mpq(s) / n for s, n in zip(inp['S'], inp['n']) if n > 0)
def M3(inp): return mp.fsum(mpq(s) for s in inp['S']) / len(inp['S'])
mutants = dict(M1=M1, M2=M2, M3=M3)
