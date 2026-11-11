"""Optional local AI reviewer. AI PASS cannot override deterministic validation failures."""
import json
from app.llm.client import json_chat

def review(text:str, data:dict, images=None):
    result=json_chat('You verify structured fields against an OCR document or images. Be skeptical. JSON only.',
     'Check whether document fields are supported. Return JSON {"status":"PASS" or "FAIL","issues":[short strings]}. '
     'Any unsupported date or license number means FAIL. Do not infer missing dates. Document OCR: '+text[:10000]+'\nExtracted data: '+json.dumps(data),images)
    if not isinstance(result,dict): return {'status':'FAIL','issues':['Reviewer returned invalid data']}
    issues=result.get('issues',[])
    if not isinstance(issues,list): issues=[str(issues)]
    status='PASS' if result.get('status')=='PASS' and not issues else 'FAIL'
    return {'status':status,'issues':[str(issue)[:180] for issue in issues[:12]]}
