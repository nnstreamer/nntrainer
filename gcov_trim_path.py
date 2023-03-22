import os
import sys

gcov_source_txt_dir = sys.argv[1]
gcov_obj_dir = sys.argv[2]

##
# @brief rename gcno files
##
def main():
    f = open(gcov_source_txt_dir + "/gcov_sources.txt", "r")

    gcov_sources_abs_lines = f.readlines()
    gcov_sources_abs = []
    for gcov_sources_abs_line in gcov_sources_abs_lines:
        gcov_sources_abs += gcov_sources_abs_line.split()
    gcov_sources = [gcov_source_abs.split('/')[-1] + ".gcno" for gcov_source_abs in gcov_sources_abs]
    gcov_sources.sort(key=len, reverse=True)

    for root, dirs, files in os.walk(gcov_obj_dir):
        for filename in files:
            for gcov_source in gcov_sources:
                if gcov_source in filename:
                    os.rename(os.path.join(root, filename), os.path.join(root, gcov_source))
                    break
                
    f.close()

if __name__ == '__main__':
    main()
