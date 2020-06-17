from argparse import ArgumentParser


def opts_parser():
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, default='linear_with_offset',
        help='dataset name (default: %(default)s)')

    parser.add_argument(
        '--svm_type', type=str, default='soft_linear_with_offset',
        help='svm_type (default: %(default)s)')

    return parser
