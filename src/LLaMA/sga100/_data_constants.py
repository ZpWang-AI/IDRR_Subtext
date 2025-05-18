from utils_zp import *
from IDRR_data import *


DATA_PATH_PDTB2_BASE = '/public/home/hongy/zpwang/IDRR_Subtext/data/used/pdtb2.p2.csv'
DATA_PATH_PDTB2_SUBTEXT = '/public/home/hongy/zpwang/IDRR_Subtext/data/subtext_distilled/pdtb2.llama3.subtext_base.csv'
DATA_PATH_PDTB3_BASE = '/public/home/hongy/zpwang/IDRR_Subtext/data/used/pdtb3.p2.csv'
DATA_PATH_PDTB3_SUBTEXT = '/public/home/hongy/zpwang/IDRR_Subtext/data/subtext_distilled/pdtb3.llama3.subtext_base.csv'

INSTRUCTION_PDTB2_BASE = '''
Argument 1:
{arg1}

Argument 2:
{arg2}

What's the discourse relation between Argument 1 and Argument 2?
A. Comparison.Concession
B. Comparison.Contrast
C. Contingency.Cause
D. Contingency.Pragmatic cause
E. Expansion.Alternative
F. Expansion.Conjunction
G. Expansion.Instantiation
H. Expansion.List
I. Expansion.Restatement
J. Temporal.Asynchronous
K. Temporal.Synchrony
'''.strip()
INSTRUCTION_PDTB3_BASE = '''
Argument 1:
{arg1}

Argument 2:
{arg2}

What's the discourse relation between Argument 1 and Argument 2?
A. Comparison.Concession
B. Comparison.Contrast
C. Comparison.Similarity
D. Contingency.Cause
E. Contingency.Condition
F. Contingency.Purpose
G. Expansion.Conjunction
H. Expansion.Equivalence
I. Expansion.Instantiation
J. Expansion.Level-of-detail
K. Expansion.Manner
L. Expansion.Substitution
M. Temporal.Asynchronous
N. Temporal.Synchronous
'''.strip()
INSTRUCTION_PDTB2_SUBTEXT = '''
Argument 1:
{arg1}

Argument 2:
{arg2}

{subtext}

What's the discourse relation between Argument 1 and Argument 2?
A. Comparison.Concession
B. Comparison.Contrast
C. Contingency.Cause
D. Contingency.Pragmatic cause
E. Expansion.Alternative
F. Expansion.Conjunction
G. Expansion.Instantiation
H. Expansion.List
I. Expansion.Restatement
J. Temporal.Asynchronous
K. Temporal.Synchrony
'''.strip()
INSTRUCTION_PDTB3_SUBTEXT = '''
Argument 1:
{arg1}

Argument 2:
{arg2}

{subtext}

What's the discourse relation between Argument 1 and Argument 2?
A. Comparison.Concession
B. Comparison.Contrast
C. Comparison.Similarity
D. Contingency.Cause
E. Contingency.Condition
F. Contingency.Purpose
G. Expansion.Conjunction
H. Expansion.Equivalence
I. Expansion.Instantiation
J. Expansion.Level-of-detail
K. Expansion.Manner
L. Expansion.Substitution
M. Temporal.Asynchronous
N. Temporal.Synchronous
'''.strip()