

class Keywords:

    def __init__(self, low_opt=None, grad=None, opt=None, opt_ts=None, hess=None, optts_block=None, sp=None):
        """
        Keywords used to specify the type and method used in electronic structure theory calculations. The input file
        for a single point calculation will look something like:

        ------------------------------------------------------------------------------------------
        <keyword line directive> autode.Keywords.sp[0] autode.Keywords.sp[1] autode.Keywords.sp[2]
        autode.Keywords.optts_block

        <coordinate directive> <charge> <multiplicity>
        .
        .
        coordinates
        .
        .
        <end of coordinate directive>
        ------------------------------------------------------------------------------------------

        :param low_opt:
        :param grad:
        :param opt:
        :param opt_ts:
        :param hess:
        :param optts_block:
        :param sp:
        :return:
        """

        self.low_opt = low_opt
        self.grad = grad
        self.opt = opt
        self.opt_ts = opt_ts
        self.hess = hess
        self.optts_block = optts_block
        self.sp = sp
