#######################################################################
####################### Change settings as needed #####################
#######################################################################
proxy_setting = True # True or False
proxy_ip = "YOUR PROXY IP" # "http://ip:port"
proxy_ip = "http://10.112.1.184:8080"
cert_path = "PATH" # input Cert Path
cert_path = "C:\\SRnD_Web_Proxy_new.crt"
#######################################################################

import os
import sys
import subprocess
import tarfile
import shutil


def set_cmd():
    cmd = "pip --trusted-host pypi.org --trusted-host files.pythonhosted.org"
    if proxy_setting: cmd += " --proxy " + proxy_ip + " --cert " + cert_path

    return [sys.executable, "-m"] + cmd.split() + ["install"]


def pip_install_requirements(requirements):
    pip_cmd = set_cmd()

    subprocess.check_call(pip_cmd + ["--upgrade", "pip"])
    for package in requirements:
        subprocess.check_call(pip_cmd + [package])


def git_clone(git_url, target_dir):
    if os.path.isdir(resource_dir): return

    from git import Repo
    Repo.clone_from(git_url, target_dir)
    print("clone success")


def unzip_openblas(openblas_path):
    openblas = tarfile.open(openblas_path + "openblas-0.3.21-x86.tar.xz")
    openblas.extractall(openblas_path)
    openblas.close()


def copy_dir(external_name, dir_name, from_path, to_path):
    if os.path.isdir(to_path + dir_name): return
    shutil.copytree(from_path + external_name, to_path + dir_name)


def set_nntrainer_project_path_setting():
    nntrainer_dir = "..\\nntrainer\\"
    external_dir = resource_dir + "external\\"
    externals = {"openblas" : "OpenBLAS-0.3.21-x86\\", 
                "mman-win32" : "mman-win32\\", 
                "iniparser" : "iniparser\\",
                "dirent" : "dirent\\",
                "dlfcn-win32" : "dlfcn-win32\\"}

    for external, dir_name in externals.items():
        if not os.path.isdir(external_dir + external): continue
        if external == "openblas" and not os.path.isdir(external_dir + "openblas\\" + dir_name):
            unzip_openblas(external_dir + "openblas\\")
        copy_dir(external, dir_name, external_dir, nntrainer_dir)


def set_api_hardlink():
    real_dir = "..\\api\\"
    api_dir = "..\\nntrainer\\api\\"
    ccapi_dir = api_dir + "ccapi\\"
    
    if os.path.isdir(api_dir): 
        shutil.rmtree(api_dir)

    os.mkdir(api_dir)
    os.mkdir(ccapi_dir)
    os.mkdir(ccapi_dir + "include")
    os.mkdir(ccapi_dir + "src")

    os.link(real_dir + "nntrainer-api-common.h", api_dir + "nntrainer-api-common.h")
    os.link(real_dir + "ccapi\\src\\factory.cpp", ccapi_dir + "src\\factory.cpp")

    files = ["common.h", "dataset.h", "layer.h", "model.h", "optimizer.h", "tensor_dim.h"]
    for f in files: 
        os.link(real_dir + "ccapi\\include\\" + f, ccapi_dir + "include\\" + f)


# nntrainer windows resource repo
git_url = "https://github.com/nnstreamer/nnstreamer-windows-resource.git"

resource_dir = ".\\nnstreamer-windows-resource\\"
requirements = ["gitpython"]

pip_install_requirements(requirements)
git_clone(git_url, resource_dir)

set_nntrainer_project_path_setting()
set_api_hardlink()