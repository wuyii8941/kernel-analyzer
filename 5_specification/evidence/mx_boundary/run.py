from pathlib import Path
from fractions import Fraction as Q
import importlib.util,sys,json,argparse
ROOT=Path(__file__).resolve().parents[3]
parser=argparse.ArgumentParser()
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
if args.out.exists():raise RuntimeError("output must be a new file")
sys.path[:0]=[str(ROOT/'5_specification'),str(ROOT/'.cache/untrusted/triton_repo_src/repo/python/triton_kernels')]
import torch,triton,types
from triton.language.target_info import cuda_capability_geq
# The local target_info wrapper imports unrelated AMD names absent in installed
# Triton. Expose exactly its sole dependency used by this CUDA kernel, pointing
# to the same installed Triton function rather than substituting a constant.
bridge=types.ModuleType("triton_kernels.target_info")
bridge.cuda_capability_geq=cuda_capability_geq
sys.modules["triton_kernels.target_info"]=bridge
from kernel_spec import mx
source=ROOT/'5_specification/sources/MX_LOCAL_TRITON_DOWNCAST.py'
spec=importlib.util.spec_from_file_location('mx_boundary_source',source)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
report={'scope':'One retained kernel on explicitly selected finite ocp_floor task; not OCP-wide conformance or a general kernel binding certificate',
 'source':str(source.relative_to(ROOT)),'kernel':'_downcast_to_mxfp','torch':torch.__version__,'triton':triton.__version__,
 'dependency_bridge':'triton_kernels.target_info exposes installed triton.language.target_info.cuda_capability_geq unchanged; unrelated AMD imports omitted',
 'device':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),
 'binding':{'input_shape':[1,32],'input_dtype':'float32','axis':1,'element_format':'e2m1','scale_format':'finite E8M0',
 'rounding_mode_argument':1,'microblock_size':32,'launch_grid':[1,1],'block_outer':32,'block_quant':128,'num_warps':4,
 'output_packed_shape':[1,16],'output_scale_shape':[1,1],'packing':'low nibble even logical coordinate, high nibble odd'},'cases':[]}
values=(Q(0),Q(1),Q(2)**-124,Q(2)**-125,Q(2)**-126,Q(2)**-127)
fp4=(Q(0),Q(1,2),Q(1),Q(3,2),Q(2),Q(3),Q(4),Q(6))
for value in values:
 x=torch.full((1,32),float(value),dtype=torch.float32,device='cuda')
 received=x.cpu().tolist()[0]
 assert all(Q(v)==value for v in received)
 expected=mx.quantize([value]*32,'e2m1',profile='ocp_floor',zero_scale=Q(2)**-127 if value==0 else None)
 runs=[]
 for initial in (0x55,0xAA):
  packed=torch.full((1,16),initial,dtype=torch.uint8,device='cuda')
  scales=torch.full((1,1),initial,dtype=torch.uint8,device='cuda')
  module._downcast_to_mxfp[(1,1)](packed,*packed.stride(),scales,*scales.stride(),x,*x.stride(),*x.shape,32,128,32,1,num_warps=4)
  torch.cuda.synchronize()
  raw=packed.cpu().tolist()[0];scale_code=int(scales.item())
  decoded=[]
  for byte in raw:
   for nibble in (byte&15,byte>>4):decoded.append((-1 if nibble&8 else 1)*fp4[nibble&7])
  scale=None if scale_code==255 else Q(2)**(scale_code-127)
  runs.append({'packed_bytes':raw,'scale_code':scale_code,'scale':str(scale),'elements':list(map(str,decoded)),
               'complete_elements_match':decoded==expected['elements'],'scale_matches':scale==expected['scale']})
 report['cases'].append({'input':str(value),'expected':{k:list(map(str,v)) if isinstance(v,list) else str(v) for k,v in expected.items()},
                         'repeat_identical':runs[0]==runs[1],'runs':runs})
out=args.out
if out.exists():raise RuntimeError('refusing to replace previous device report')
out.write_text(json.dumps(report,indent=2)+'\n')
print(report['device'],report['triton'])
for case in report['cases']:print(case['input'],case['repeat_identical'],case['runs'][0]['scale_matches'],case['runs'][0]['complete_elements_match'])
