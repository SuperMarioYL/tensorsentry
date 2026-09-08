import json,struct,tempfile
from pathlib import Path
from tensorsentry.safetensors_reader import read_header
from tensorsentry.tensor_validate import TensorProfile,TensorSpec,validate_structure
profile=TensorProfile(model_id='demo-linear',architecture='linear',required_tensors=(TensorSpec(logical_name='weight',pattern='weight',dtype='F32',shape_constraints={'ndim':1,'dim0_eq':2}),))
header=json.dumps({'weight':{'dtype':'F32','shape':[2],'data_offsets':[0,8]}}).encode()
with tempfile.TemporaryDirectory() as tmp:
 path=Path(tmp)/'demo.safetensors';path.write_bytes(struct.pack('<Q',len(header))+header+struct.pack('<ff',1.0,2.0))
 tensors=read_header(str(path));good=validate_structure(tensors,profile);missing=validate_structure([],profile)
 print(json.dumps({'fixture_profile':profile.model_id,'tensor_count':good.n_tensors,'valid_structure':good.structure,'empty_structure':missing.structure,'missing_codes':[x.code for x in missing.anomalies]},indent=2))
