#!/bin/env python3
#
# Copyright 2019 Associated Universities, Inc. Washington DC, USA.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import argparse
import itertools
import re
import subprocess
import sys
import tempfile

def named_result(ln):
    end_state = ln.find('\t')
    end_name = ln.rfind('\t')
    if end_name == end_state:
        end_name = -1
    if end_name != -1:
        return (ln[end_state + 2:end_name], ln[0:end_state])
    else:
        return (ln[end_state + 2:], ln[0:end_state])

def run_test(compare_with, compare_by, test, *args):
    with tempfile.TemporaryFile('w+t') as out:
        cp = subprocess.run(
            args=[test] + list(args), stdout=out.fileno(), stderr=out.fileno())
        if cp.returncode != 0:
            sys.exit(cp.returncode)
        out.seek(0)
        log = [l for ln in out
               for l in [ln.strip()]
               if len(l) > 0
               and re.match("PASS|FAIL|SKIPPED", l) is not None]
        if compare_with is None:
            fails = map(lambda lg: lg[0:lg.find('\t')] == 'FAIL', log)
            sys.exit(1 if any(fails) else 0)
        elif compare_by == 'line':
            eq = map(lambda lns: lns[0].strip() == lns[1],
                     itertools.zip_longest(compare_with, log, fillvalue=''))
            sys.exit(0 if all(eq) else 1)
        else:
            results = dict(named_result(lg) for lg in log)
            expected = dict(named_result(l) for ln in compare_with
                            for l in [ln.strip()]
                            if len(l) > 0)
            eq_keys = set(results.keys()) == set(expected.keys())
            eq = map(lambda k: k in expected and expected[k] == results[k],
                     results.keys())
            sys.exit(0 if eq_keys and all(eq) else 1)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run a TestLog-based test program')
    parser.add_argument(
        '--compare-with',
        nargs='?',
        type=argparse.FileType('r'),
        help='file with expected output',
        metavar='FILE',
        dest='compare_with')
    parser.add_argument(
        '--compare-by',
        choices=['line', 'name'],
        nargs='?',
        default='name',
        help='compare by line number or test name (default "%(default)s")',
        metavar='line|name',
        dest='compare_by')
    parser.add_argument(
        'test',
        nargs=1,
        type=argparse.FileType('r'),
        help='test executable name',
        metavar='TEST')
    parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        help='test arguments',
        metavar='ARGS')
    args = parser.parse_args()
    if 'compare_with' in args:
        compare_with = args.compare_with
    else:
        compare_with = None
    test_name = args.test[0].name
    args.test[0].close()
    run_test(compare_with, args.compare_by, test_name, *args.args)
