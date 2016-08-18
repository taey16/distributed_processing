#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pprint import pprint

#import pdb; pdb.set_trace()
with open('texture_detector/part-00000.json') as data_file:    
  for line in data_file:
    data = json.loads(line)

pprint(data)
