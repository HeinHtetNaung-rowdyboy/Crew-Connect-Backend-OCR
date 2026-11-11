from typing import Optional
from pydantic import BaseModel, Field

class CloudFields(BaseModel):
    training_date: Optional[str] = None
    issue_date: Optional[str] = None
    examination_date: Optional[str] = None
    expire_date: Optional[str] = None
    license_number: Optional[str] = None

class CertificateData(CloudFields):
    certificate_name: str
    certificate_type: Optional[str] = None
    category: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    expiration: str = 'None'
