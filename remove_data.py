import optparse
import os
import pandas as pd
import sys


def parse_args(parser):
    options, args = parser.parse_args()
    return options


def add_parser():
    parser = optparse.OptionParser()

    parser.add_option('-t', '--persontag', action="store", dest="tag",
                      help="tag of a person to be deleted from the database",
                      default="klaudia")
    parser.add_option('-f', '--datafile', action="store", dest="file",
                      help="csv file containing recognition data",
                      default="data_personal.csv")
    return parser


def delete(*args, **kwargs):
    print("Parsing data...")
    parser = add_parser()
    args = parse_args(parser)

    if not os.path.isfile(args.file):
        print("{} does not exist.".format(args.file))
        return
    print("Removing data ({})...".format(args.tag))
    file_tmp = "tmp.csv"
    data = pd.read_csv(args.file)
    data = data[data["tag"] != args.tag]
    data.to_csv(path_or_buf=file_tmp, index=False)

    os.remove(args.file)
    os.rename(file_tmp, args.file)

    print("Done.")
    return


if __name__ == "__main__":
    delete(sys.argv[1:])
