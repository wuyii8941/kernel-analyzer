"""Maps the families of the library survey (survey_9bb8abe/classes.json) that needed a definition onto the v2.5
contracts by keyword rules on the family names, and lists what stays open. This batch was used to design v2.5, so the
numbers describe where v2.5 stands on it; they are not a first-contact measurement (the next fresh batch is)."""
import os, json, re, collections
HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(HERE, 'survey_9bb8abe', 'classes.json')))
RULES = {   # class -> ordered (regex on the lower-cased family name, contract or 'open: reason')
 'A': [(r'segment', 'A1_LINEAR'), (r'rwkv|hgrn|abc\b|based|rebased|\bfox\b|forgetting|path|mesa|log.?linear|comba|ttt|titans|deltaproduct|mom\b|hope', 'open: variant not in A1 (RWKV, HGRN, ABC, Based, FoX, PaTH, ...)'),
       (r'conv|causal_conv|short.?conv|shift', 'open: convolution state (A1.D11 / D7)'),
       (r'kda|kimi', 'A1_LINEAR'), (r'gated.?delta|gdn', 'A1_LINEAR'), (r'delta', 'A1_LINEAR'),
       (r'\bgla\b|gated linear|gsa', 'A1_LINEAR'), (r'retention|retnet|simple.?gla|lightning|mamba|ssd|ssm|linear.?attn|linear attention|decay', 'A1_LINEAR'),
       (r'common|chunk|cumsum|wy|solve.?tril|ut.?transform|state|recurrent|fused.?recurrent|l2.?norm|gate|beta', 'A1_LINEAR')],
 'B': [(r'mla|absorb|latent', 'open: MLA absorbed form'), (r'quant|int8|int4|fp8|fp4', 'open: quantized KV target (B1.D8)'),
       (r'.', 'B1_PAGED')],
 'C': [(r'compress|cmp|ring|window.?compress', 'open: compressed KV and ring buffers (C1.D7)'), (r'fp8|fp4|quant|ue8m0|byte', 'open: KV byte layouts (C1.D5)'),
       (r'index|topk|top-k|select|nsa|dsa|score|sparse|block', 'C1_SPARSE')],
 'E': [(r'fp8', 'F7_FP8SCALE + B1_PAGED'), (r'int8|int4|quant|scale|zero', 'F4_INTQ + B1_PAGED'), (r'.', 'B1_PAGED')],
 'F': [(r'gate', 'open: not a quantization kernel (gating)'), (r'dynamic.?quant', 'v2.4 NORM + F7_FP8SCALE / F4_INTQ'), (r'nvfp4|fp4.*16', 'F3_NVFP4'), (r'mx|e8m0|ue8m0|microscal', 'F2_MX'), (r'int8|int4|w8a8|w4|zero.?point|asym', 'F4_INTQ'),
       (r'fp8|e4m3|e5m2|per.?token|per.?tensor|per.?block|group|1x128|128x128|act.?quant', 'F7_FP8SCALE'), (r'.', 'open: format not matched')],
 'G': [(r'mx|nvfp4|dot_scaled|e8m0', 'F2_MX / F3_NVFP4 + MATMUL'), (r'awq|gptq|pack|w4a16|int4|w4|w8|marlin|gemm', 'F5_PACK + F4_INTQ'), (r'.', 'open: format not matched')],
 'H': [(r'quant|fp8|int8|int4|layout|swizzle', 'open: quantized MoE layouts (H2.D5)'), (r'gptq|awq|w4a16', 'H2_ALIGN + F5_PACK'), (r'dispatch|combine|all.?to.?all|a2a|\bep\b|deepep', 'K14_COLL'), (r'align|sort|permute|scatter|gather|count', 'H2_ALIGN'),
       (r'rout|topk|top-k|gate|gating|softmax|sigmoid|bias|group', 'H1_ROUTING'), (r'moe|expert|gemm|fused', 'H2_ALIGN'), (r'.', 'open: not matched')],
 'I': [(r'hash|murmur|watermark|spec|verify|tree|accept|draft|eagle', 'open: speculative decoding / hashing / watermark'), (r'sampl|top.?p|top.?k|penalt|temperature|min.?p', 'R1_RANDOM (D2 open)'), (r'.', 'open: speculative decoding / hashing / watermark')],
 'J': [(r'.', 'open: layout and metadata formats')],
 'K': [(r'.', 'open: block-sparse metadata')],
 'L': [(r'.', 'open: LoRA layouts')],
 'M': [(r'mrope|multimodal|vision|vl', 'open: mrope positions'), (r'.', 'v2.4 ROPE role')],
 'O': [(r'.', 'open: other')],
 'R': [(r'dropout', 'R1_RANDOM'), (r'sampl', 'R1_RANDOM (D2 open)'), (r'.', 'open: other randomness')],
}
rows, per_class = [], {}
for cls, info in C.items():
    cnt = collections.Counter(); unmatched = []
    for fam in info['families']:
        name = fam['family'].lower(); target = None
        for rx, tgt in RULES.get(cls, []):
            if re.search(rx, name): target = tgt; break
        target = target or 'open: not matched'
        cnt['open' if target.startswith('open') else 'mapped'] += 1
        rows.append((cls, fam['library'], fam['family'], target))
    per_class[cls] = (info['name'], len(info['families']), cnt['mapped'], cnt['open'])
tot = sum(v[1] for v in per_class.values()); mapped = sum(v[2] for v in per_class.values())
with open(os.path.join(HERE, 'COVERAGE_v25.md'), 'w') as f:
    f.write('# v2.5 on the 15-library survey (families that needed a definition or are random)\n\n')
    f.write('This batch was used to design v2.5; the numbers below say where v2.5 stands on it and are not a first-contact measurement.\n\n')
    f.write('| class | name | families | mapped to a v2.5 contract | still open |\n|---|---|---|---|---|\n')
    for cls, (nm, n, m, o) in per_class.items(): f.write(f'| {cls} | {nm} | {n} | {m} | {o} |\n')
    f.write(f'| all | | {tot} | {mapped} ({100*mapped/tot:.0f}%) | {tot-mapped} |\n\n')
    f.write('Before v2.5 (first contact), 453 of 797 families had a reference program (torch operation or composition) and 344 needed a definition or were random.\n\n')
    f.write('## Per family\n\n| class | library | family | v2.5 contract or open item |\n|---|---|---|---|\n')
    for r in rows: f.write('| ' + ' | '.join(r) + ' |\n')
print('| class | name | families | mapped | open |')
for cls, (nm, n, m, o) in per_class.items(): print(f'| {cls} | {nm} | {n} | {m} | {o} |')
print(f'TOTAL {tot} families: mapped {mapped} ({100*mapped/tot:.0f}%), open {tot-mapped}')
print('open reasons:', collections.Counter(r[3] for r in rows if r[3].startswith('open')).most_common())
