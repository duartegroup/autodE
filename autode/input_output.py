from .log import logger


def xyzs2xyzfile(xyzs, filename=None, basename=None, title_line=''):
    """
    For a list of xyzs in the form e.g [[C, 0.0, 0.0, 0.0], ...] convert create a standard .xyz file

    :param xyzs: List of xyzs
    :param filename: Name of the generated xyz file
    :param basename: Name of the generated xyz file without the file extension
    :param title_line: String to print on the title line of an xyz file
    :return: The filename
    """

    if basename:
        filename = basename + '.xyz'

    if filename is None:
        logger.error('xyz filename cannot be None')
        return 1

    logger.info('Generating xyz file for {}'.format(filename))

    if filename.endswith('.xyz'):
        with open(filename, 'w') as xyz_file:
            if xyzs:
                print(len(xyzs), '\n', title_line, sep='', file=xyz_file)
            else:
                logger.error(' No xyzs to print')
                return 1
            [print('{:<3}{:^10.5f}{:^10.5f}{:^10.5f}'.format(*line), file=xyz_file) for line in xyzs]

    return filename
