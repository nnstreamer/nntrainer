"""
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.

@file build_windows_resources.py
@date 16 July 2025
@brief Extract symbols from obj files to build def file
@author Grzegorz Kisala <gkisala@gmail.com>
"""
import sys
import os
import re
import subprocess
import argparse
from collections import defaultdict

def extract_syms(symfile, defs):
    print('Opening ', symfile)
    with open(symfile, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            line = re.sub(r'notype \(\)', 'func', line)
            line = re.sub(r'notype', 'data', line)
            pieces = line.split()
            
            if not pieces[0] or not re.match('^[A-F0-9]{3,}$', pieces[0]):
                continue

            try:
                symbol_name = pieces[6]
            except IndexError:
                continue
            
            if pieces[2] == "UNDEF":
                continue
            
            if pieces[4] != "External":
                continue
            
            if re.match('^@', symbol_name) or re.match('^[(]', symbol_name):
                continue
            
            if re.match('^__real', symbol_name) or re.match('^__xmm', symbol_name):
                continue
            
            if re.match('^__imp', symbol_name):
                continue
            
            if re.search('NULL_THUNK_DATA$', symbol_name) or re.match('^__IMPORT_DESCRIPTOR', symbol_name) or re.match('^__NULL_IMPORT', symbol_name):
                continue
            
            if re.match('^\\?\\?_C', symbol_name):
                continue
            
            defs[symbol_name] = pieces[3]

def writedef(deffile, defs):
    with open(deffile, 'w') as fh:
        fh.write("EXPORTS\n")
        for k in sorted(defs.keys()):
            isdata = defs[k] == 'data'

            if isdata:
                fh.write("  {} DATA\n".format(k))
            else:
                fh.write("  {}\n".format(k))

REPO_DIR = os.path.abspath(os.path.dirname(__file__))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects_dir', type=str, required=True, help='Path to directory with obj files')
    parser.add_argument('--working_dir', type=str, required=True, help='Path to working directory')
    return parser.parse_args()

def run_command(args, cwd, exit_on_failure=True):
    cmd = ' '.join(args)

    print('Running command: {}'.format(cmd))

    returncode = subprocess.run(args, cwd=cwd, check=exit_on_failure).returncode

    if exit_on_failure and returncode != 0:
        print('Application {} failed with returncode {}. Exiting...'.format(
            args[0], returncode))
        sys.exit(returncode)

    return returncode

def list_files(path, extension):
    file_list = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path) and file_path.endswith(extension):
            file_list.append(file_path)
    
    return file_list

def main():
    args = get_args()

    print("args.objects_dir: {}".format(args.objects_dir))
    sym_file = os.path.join(args.working_dir, 'nntrainer.sym')
    def_file = os.path.join(args.working_dir, 'nntrainer.def')
    files = list_files(args.objects_dir, '.obj')

    dumpbin_command = ['dumpbin', '/symbols', '/out:{}'.format(sym_file)] + files

    run_command(dumpbin_command, REPO_DIR)

    defs = defaultdict()
    extract_syms(sym_file, defs)

    writedef(def_file, defs)

if __name__ == '__main__':
    main()
