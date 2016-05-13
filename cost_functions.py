def lms_f0(x):
    pass


def lms_f1(x):
    pass


def lmls_f0(x):
    pass


def lmls_f1(x):
    pass


cost_functions_set = {}
cost_functions_set['LMS'] = lms_f0
cost_functions_set['LMLS'] = lmls_f0

cost_functions_derivative_set = {}
cost_functions_derivative_set['LMS'] = lms_f1
cost_functions_derivative_set['LMLS'] = lmls_f1
