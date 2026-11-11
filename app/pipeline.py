from pathlib import Path
from app.config import settings
from app.ocr.pdf_parser import parse_pdf as local_parse
from app.ocr.landingai import parse_pdf as cloud_parse, extract_fields as cloud_extract
from app.classification.classifier import classify
from app.extraction.extractor import extract_fields
from app.validation.validator import validate_and_normalize

def process_document(path:Path, use_ollama:bool|None=None)->dict:
    cfg=settings(); mode=cfg['backend']; warnings=[]
    if mode not in ('local','landingai','hybrid'): raise ValueError('Invalid OCR_BACKEND')
    parsed=cloud_parse(path) if mode=='landingai' else local_parse(path)
    classification=classify(parsed['text'])
    fields=extract_fields(parsed['text'],classification['certificate_name'])
    if mode=='hybrid' and classification['requires_review'] and cfg['landingai_key']:
        try:
            parsed=cloud_parse(path);classification=classify(parsed['text']); fields=extract_fields(parsed['text'],classification['certificate_name'])
        except Exception as exc: warnings.append('Cloud fallback failed: '+type(exc).__name__)
    if cfg['use_ai_extraction'] and parsed['engine']=='landingai':
        try:
            supplements=cloud_extract(parsed['text'])
            for k in fields:
                if not fields[k] and supplements.get(k):fields[k]=supplements[k]
        except Exception as exc: warnings.append('Cloud extraction failed: '+type(exc).__name__)
    ai_enabled=cfg['use_ollama'] if use_ollama is None else use_ollama
    reviewer={'status':'NOT_RUN','issues':[]}
    if ai_enabled:
        from app.extraction.vision import extract_vision,render_pages
        imgs=None
        try:
            imgs=render_pages(path)
            if cfg['use_llm_classifier'] and classification['requires_review']:
                from app.classification.llm_classifier import llm_classify
                proposal=llm_classify(parsed['text'],imgs)
                if proposal['certificate_name']!='Unknown':classification=proposal
            prediction=extract_vision(path,parsed['text'],classification['certificate_name'],imgs)
            # Vision may correct detected values: human verification is still required.
            for key,value in prediction.items():
                if value: fields[key]=value
        except Exception as exc: warnings.append('Ollama extraction failed; fallback to OCR rules: '+type(exc).__name__)
    data,validation_warnings=validate_and_normalize(classification,fields)
    warnings.extend(validation_warnings)
    if ai_enabled and cfg['use_ai_reviewer']:
        try:
            from app.validation.reviewer import review
            reviewer=review(parsed['text'],data,imgs)
        except Exception as exc: reviewer={'status':'FAIL','issues':['AI review unavailable: '+type(exc).__name__]}
    deterministic_fail=bool(validation_warnings) or classification['requires_review']
    if reviewer['status']=='PASS' and deterministic_fail:
        reviewer={'status':'FAIL','issues':reviewer['issues']+['Rule validation requires review']}
    if not ai_enabled: reviewer={'status':'NOT_RUN','issues':['Enable Qwen/Ollama to use AI reviewer']}
    needs_review=deterministic_fail or reviewer['status']=='FAIL' or (ai_enabled and reviewer['status']!='PASS')
    return {'success':True,'data':data,'classification':{'confidence':classification['confidence'],'matched_aliases':classification['matched_aliases']},'validation':reviewer, 'requires_review':needs_review,'warnings':warnings,'processing':{'engine':parsed['engine'],'page_count':len(parsed['pages']),'qwen_enabled':ai_enabled,'ocr_engines_used':parsed.get('ocr_engines_used',[])}}
