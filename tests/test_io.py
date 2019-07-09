import os
from autode import input_output

here = os.path.dirname(os.path.abspath(__file__))


def test_xyz_file_generation():
    os.chdir(os.path.join(here, 'input_output'))

    xyzs = [['H', 0.0, 0.0, 0.0], ['H', 1.0, 0.0, 0.0]]
    input_output.xyzs2xyzfile(xyzs=xyzs, basename='xyzfile1')
    assert os.path.exists('xyzfile1.xyz')

    xyz_file_lines = open('xyzfile1.xyz', 'r').readlines()
    assert len(xyz_file_lines) == 4
    assert int(xyz_file_lines[0].split()[0]) == len(xyzs)

    input_output.xyzs2xyzfile(xyzs=xyzs, filename='xyzfile2.xyz')
    assert os.path.exists('xyzfile2.xyz')
