"""Job-title → sector mapping shared by notebooks, retraining and the released artefact.

Copied verbatim from `Notebooks/02_Calidad de Datos.ipynb` (section "Categorización de empleo").
Categories are evaluated in dictionary order and the first match wins, so specific professions come
first and seniority words (`direccion`) last: "nurse manager" -> salud, "sales manager" -> ventas.
"""

import pandas as pd

# Order matters: the first matching sector wins.
PATRONES_SECTOR = {
    'legal': ['attorney', 'lawyer', 'paralegal', 'legal', 'counsel', 'judge'],
    'salud': ['nurse', 'nursing', 'rn', 'lpn', 'lvn', 'cna', 'physician', 'doctor', 'md',
              'medical', 'pharmacist', 'pharmacy', 'therapist', 'dental', 'dentist',
              'hygienist', 'clinical', 'hospital', 'patient', 'caregiver', 'care giver',
              'health', 'healthcare', 'surgical', 'surgeon', 'radiologic', 'radiology',
              'paramedic', 'emt', 'phlebotomist', 'veterinarian', 'veterinary', 'lab',
              'laboratory', 'technologist', 'optometrist', 'optician', 'chiropractor',
              'pathologist'],
    'educacion': ['teacher', 'professor', 'school', 'instructor', 'education', 'educator',
                  'tutor', 'faculty', 'librarian', 'university', 'college',
                  'paraprofessional', 'trainer'],
    'finanzas': ['accountant', 'accounting', 'bookkeeper', 'financial', 'finance', 'banker',
                 'bank', 'loan', 'mortgage', 'underwriter', 'auditor', 'controller', 'tax',
                 'payroll', 'credit', 'insurance', 'claims', 'billing', 'teller', 'cfo',
                 'cpa', 'investment', 'collector', 'collections', 'accounts payable',
                 'accounts receivable', 'compliance'],
    # 'officer' alone is excluded: it would clash with "chief executive officer" or "loan officer"
    'seguridad_publica': ['police', 'sheriff', 'deputy', 'correctional', 'corrections',
                          'correction officer', 'probation', 'firefighter', 'fire',
                          'sergeant', 'lieutenant', 'captain', 'trooper', 'detective',
                          'investigator', 'special agent', 'law enforcement', 'military',
                          'army', 'navy', 'air force', 'marine', 'usps', 'postal',
                          'mail carrier', 'letter carrier', 'city carrier', 'rural carrier',
                          'federal', 'government', 'county', 'city of', 'state of',
                          'security'],
    'tecnologia': ['software', 'developer', 'programmer', 'it', 'network', 'systems',
                   'system', 'data', 'web', 'database', 'engineer', 'engineering',
                   'architect', 'devops', 'computer', 'technology', 'desktop', 'helpdesk',
                   'scientist', 'chemist'],
    'transporte': ['driver', 'truck', 'trucker', 'bus', 'pilot', 'flight', 'dispatcher',
                   'warehouse', 'logistics', 'delivery', 'courier', 'forklift', 'shipping',
                   'freight', 'transportation', 'conductor', 'carrier', 'mail handler',
                   'mailhandler', 'material handler', 'packer'],
    'hosteleria': ['server', 'bartender', 'cook', 'chef', 'food', 'restaurant', 'waiter',
                   'waitress', 'hotel', 'housekeeper', 'housekeeping', 'barista',
                   'kitchen', 'catering'],
    'oficios': ['mechanic', 'electrician', 'welder', 'carpenter', 'machinist', 'operator',
                'laborer', 'labor', 'foreman', 'forman', 'maintenance', 'technician', 'tech',
                'construction', 'plumber', 'installer', 'production', 'plant', 'assembler',
                'assembly', 'hvac', 'painter', 'lineman', 'custodian', 'janitor',
                'fabricator', 'manufacturing', 'mason', 'inspector', 'millwright',
                'pressman', 'contractor', 'quality control', 'quality assurance'],
    'ventas': ['sales', 'salesman', 'retail', 'cashier', 'store', 'realtor', 'real estate',
               'merchandiser', 'broker', 'buyer', 'marketing', 'service advisor'],
    'administrativo': ['assistant', 'secretary', 'receptionist', 'clerk', 'clerical',
                       'administrative', 'admin', 'administrator', 'administration',
                       'office', 'customer service', 'csr', 'coordinator', 'hr',
                       'human resources', 'recruiter', 'processor', 'representative', 'rep',
                       'specialist', 'analyst', 'associate', 'consultant', 'advisor',
                       'planner', 'estimator'],
    'direccion': ['manager', 'mgr', 'gm', 'management', 'director', 'owner', 'president',
                  'ceo', 'coo', 'cto', 'vp', 'executive', 'supervisor', 'partner',
                  'superintendent', 'chief', 'lead', 'leader', 'principal', 'founder',
                  'head'],
}

# Every value `asignar_sector` can return; the API contract accepts exactly these.
SECTORES_VALIDOS = [*PATRONES_SECTOR, 'otros', 'desconocido']

# One regex per sector; \b forces whole-word matches ("rn" must not match inside "intern").
REGEX_SECTOR = {sector: r'\b(?:' + '|'.join(palabras) + r')\b'
                for sector, palabras in PATRONES_SECTOR.items()}


def normalizar_empleo(serie: pd.Series) -> pd.Series:
    """Lowercase, drop punctuation and collapse spaces ("Registered Nurse," -> "registered nurse")."""
    return (serie.astype('string').str.lower()
                 .str.replace(r'[^a-z0-9 ]', ' ', regex=True)
                 .str.replace(r'\s+', ' ', regex=True)
                 .str.strip())


def asignar_sector(serie: pd.Series) -> pd.Series:
    """Map raw job titles to a sector: first matching pattern, 'otros' if none, 'desconocido' if missing."""
    empleo_norm = normalizar_empleo(serie)
    coincidencias = pd.DataFrame({sector: empleo_norm.str.contains(regex, regex=True).fillna(False).astype(bool)
                                  for sector, regex in REGEX_SECTOR.items()}, index=serie.index)
    sector = coincidencias.idxmax(axis=1).where(coincidencias.any(axis=1), 'otros')
    return sector.mask(empleo_norm.isna(), 'desconocido').astype(object)
