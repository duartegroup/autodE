from autode.log import logger


def xyzs2xyzfile(xyzs, filename=None, basename=None, title_line=''):
    """For a list of xyzs in the form e.g [[C, 0.0, 0.0, 0.0], ...] create a standard .xyz file

    Arguments:
        xyzs {list(list)} -- List of xyzs

    Keyword Arguments:
        filename {str} -- Name of the generated xyz file (default: {None})
        basename {str} -- Name of the generated xyz file without the file extension (default: {None})
        title_line {str} -- String to print on the title line of an xyz file (default: {''})

    Returns:
        {str} -- Name of the generated xyz file
    """

    if basename:
        filename = basename + '.xyz'

    if filename is None:
        logger.error('xyz filename cannot be None')
        return None

    logger.info(f'Generating xyz file for {filename}')

    if not filename.endswith('.xyz'):
        logger.error('xyz filename does not end with .xyz')
        return None

    if xyzs is None:
        logger.error(' No xyzs to print')
        return None

    if len(xyzs) == 0:
        logger.error(' No xyzs to print')
        return None

    with open(filename, 'w') as xyz_file:
        print(len(xyzs), '\n', title_line, sep='', file=xyz_file)
        [print('{:<3} {:^10.5f} {:^10.5f} {:^10.5f}'.format(
            *line), file=xyz_file) for line in xyzs]

    return filename
