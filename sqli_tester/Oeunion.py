#!/usr/bin/env python

"""
Copyright (c) 2006-2022 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
# had to put it into this file because Python won't let me import it.
import re

# from lib.core.enums import PRIORITY

# __priority__ = PRIORITY.HIGHEST

class Oeunion:
	def __init__(self):
                pass
	def dependencies():
	    pass

	def tamper(payload, **kwargs):
	    """
    Replaces instances of <int> UNION with <int>e0UNION

    Requirement:
        * MySQL
        * MsSQL

    Notes:
        * Reference: https://media.blackhat.com/us-13/US-13-Salgado-SQLi-Optimization-and-Obfuscation-Techniques-Slides.pdf

    >>> tamper('1 UNION ALL SELECT')
    '1e0UNION ALL SELECT'
    """

	    return re.sub(r"(?i)(\d+)\s+(UNION )", r"\g<1>e0\g<2>", payload) if payload else payload