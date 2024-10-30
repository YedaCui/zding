#! /bin/bash

ls solver/*.py | cut -d "." -f 1 | cut -d "/" -f 2 | xargs -n1 -P64 -I{} -t bash -c "timeout 600 python3.8 -u test.py {} > log/{}.txt"
python3.8 report.py
