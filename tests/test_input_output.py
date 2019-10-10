from autode import input_output


xyz_list = [['H', 0.0, 0.0, 0.0], ['H', 1.0, 0.0, 0.0]]


def test_xyzsxyzfile():

    no_name = input_output.xyzs2xyzfile(xyz_list, None, None,)
    assert no_name == None

    wrong_ending = input_output.xyzs2xyzfile(xyz_list, 'test.abc', None,)
    assert wrong_ending == None

    xyz_empty = []
    no_xyz = input_output.xyzs2xyzfile(xyz_empty, None, 'test',)
    assert no_xyz == None

    no_xyz2 = input_output.xyzs2xyzfile(None, None, 'test',)
    assert no_xyz2 == None
