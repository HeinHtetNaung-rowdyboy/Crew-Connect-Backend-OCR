from app.classification.classifier import load_certificate_config
from app.llm.client import json_chat

def llm_classify(text:str,images=None):
    definitions=load_certificate_config()
    names=[x['name'] for x in definitions]
    answer=json_chat('You classify maritime documents. Reply with JSON containing certificate_name, evidence and confidence (0-1). Do not invent a name.',
      'Choose exactly one certificate_name from '+str(names)+' or Unknown. Use document title and content, not filenames. OCR text: '+text[:10000],images)
    name=str(answer.get('certificate_name','Unknown'))
    match=next((x for x in definitions if x['name'].casefold()==name.casefold()),None)
    if not match:
        return {'certificate_name':'Unknown','certificate_type':None,'category':None,'confidence':0,'matched_aliases':[],'requires_review':True}
    try: confidence=float(answer.get('confidence',0))
    except (TypeError,ValueError): confidence=0
    confidence=min(max(confidence,0),1)
    return {'certificate_name':match['name'],'certificate_type':match['certificate_type'],'category':match['category'],'confidence':round(confidence,2), 'matched_aliases':[str(answer.get('evidence','LLM'))[:120]],'requires_review':confidence<0.8}
