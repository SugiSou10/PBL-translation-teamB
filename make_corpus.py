import sys

args = sys.argv

def cut(fname, f_en, f_ja):
    fin = open(fname, "r")
    f1 = open(f_en, "w")
    f2 = open(f_ja, "w")
    for line in fin:
        part = line.strip().split("\t")
        f1.write(part[3] + "\n")
        f2.write(part[4] + "\n")       
    fin.close()
    f1.close()
    f2.close()

if __name__ == '__main__':
    cut(args[1], args[2], args[3])