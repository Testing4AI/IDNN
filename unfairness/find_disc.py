sens_param = 8 # 8 for race
conf = census

import copy

# def get_random_input(conf, sens_param):
#     x = []
#     for i in range(conf.params):
#         x.append(random.randint(conf.input_bounds[i][0], conf.input_bounds[i][1]))
#     x[sens_param - 1] = 0
#     return x


# def get_estimate(model, conf, r_samples, sens_param):
#     disc_count = 0
#     total_count = r_samples.shape[0]
#     for inp in r_samples:
#         if check_for_error_condition_x(model, conf, inp, sens_param):
#             disc_count = disc_count + 1

#     estimate = 1 - float(disc_count)/total_count
#     return estimate


def check_for_error_condition_x(model, conf, t, sens):
    t = np.array(t).astype("int")
    label = np.argmax(model(np.array([t])), axis=1)

    # check for all the possible values of sensitive feature
    for val in range(conf.input_bounds[sens-1][0], conf.input_bounds[sens-1][1]+1):
        if val != int(t[sens-1]):
            tnew = copy.deepcopy(t)
            tnew[sens-1] = val
            label_new = np.argmax(model(np.array([tnew])), axis=1)
            if label_new != label:
                return tnew
    return None

def find_discs(model, conf, totals, sens_param):
    oldlist = []
    ttlist = []
    for inp in totals:
        tnew = check_for_error_condition_x(model, conf, inp, sens_param)
        if tnew is None:
            continue
        if tnew.any():
            oldlist.append(inp)
            ttlist.append(tnew)
    return np.array(oldlist), np.array(ttlist)