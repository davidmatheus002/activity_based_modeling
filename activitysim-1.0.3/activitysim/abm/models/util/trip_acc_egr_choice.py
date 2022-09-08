import pandas as pd


def combine_mode_ids(acc_id, main_id, egr_id):
    """
    construct a integer to combine the access, main and egress mode id's 0~2 digits egress, 3~5 main, 6~8 access
    :param acc_id:
    :param main_id:
    :param egr_id:
    :return:
    """
    return int(acc_id * 10**6 + main_id * 10**3 + egr_id)


def split_combined_id(val):
    """
    :param val: it is a combined mode id of access, main and egress
    :return: access mode id, main mode id and egress id in a tuple
    """
    tmp = 10**6
    return val // tmp, (val % tmp) // 10**3, val % 10**3


def convert_mode_name2id(name, mode_name2id_map):
    """
    convert the (uni/multi-modal)mode name to a combined integer
    :param name:
    :param mode_id_map:
    :return:
    """
    pos = name.find('_')
    pos_next = name.rfind('_')
    if pos_next > pos > 0:
        access_mode_id = mode_name2id_map[name[0: pos]]
        egress_mode_id = mode_name2id_map[name[pos_next + 1:]]
        main_mode_id = mode_name2id_map[name[pos + 1: pos_next]]
    else:
        access_mode_id = egress_mode_id = 0
        main_mode_id = mode_name2id_map[name]
    return combine_mode_ids(access_mode_id, main_mode_id, egress_mode_id)


def convert_mode_id2name(val, mode_id2name_map):
    if val is None:
        return "-"
    rtn = split_combined_id(val)
    # access at 0 pos, main at 1 pos, egress at 2 pos
    if rtn[0] != 0 and rtn[2] != 0:
        return mode_id2name_map[rtn[0]] + '_' + mode_id2name_map[rtn[1]] + '_' + mode_id2name_map[rtn[2]]
    else:
        return mode_id2name_map[rtn[1]]


def extract_mode_name(orig_name):
    """
    Extract the multi-modal name from names like walk_car_bike_ACC, which should be walk_car_bike.
    :param orig_name:
    :return: valid name
    """
    if orig_name[-4:].upper() in ["_ACC", "_EGR"]:
        return orig_name[:-4].lower()
    elif orig_name[-2:].upper() == "_M":
        return orig_name[:-2].lower()
    else:
        return orig_name.lower()


def create_ownership_number(has_car, has_bike, has_ebike):
    """
    construct an integer to represent the travel resource ownership, where the
    1st bit represents car ownership, 2nd bit represent bike ownership, 3rd bit represents ebike-ownership
    :param has_car: it is the combination of car-ownership with driving license
    :param has_bike:
    :param has_ebike:
    :return:
    """
    a = int('000000000000', 2)
    a |= has_car << 0   # The 1st bit indicates whether the agent owns a car or not;
    a |= has_bike << 1
    a |= has_ebike << 2
    return a


def get_output_filename(has_car, has_bike, has_ebike, maas_enabled, total_num_modes):
    """
    As we need to save the mode chain sets in a HDF5 file for different type of agent,
    this function will create a output file name based on the properties of agent. Since each agent sample contains
    the ownership information. We use their bit composition to represent. E.g

    agent_id HAS_CAR  HAS_BIKE HAS_EBIKE        INTERGER
      1          0       0        0               0
      2          1       0        0               1
      ..
    The 1st bit indicates whether the agent owns a car or not;
    The 2nd bit .................................a bike or not; And so on..
    :param has_car:
    :param has_bike:
    :param has_ebike:
    :param maas_enabled: indicate if MaaS is enabled
    :param total_num_modes
    :return: file name
    """
    a = create_ownership_number(has_car, has_bike, has_ebike)
    if maas_enabled:
        return "ModeChainSet_MaaS_" + str(total_num_modes) + '_TotalModes.h5'
    else:
        return 'ModeChainSet_' + str(a) + '_ownership_' + str(total_num_modes) + '_TotalModes.h5'


def read_modes_from_settings(model_settings):
    """
    Return a list of modes specified in the YAML file (which is also specified in CSV), the name are lower case!
    :param model_settings:
    :return:
    """
    logit_type = model_settings.get('LOGIT_TYPE')
    if logit_type == "MNL":
        mode_spec = pd.DataFrame.from_records(model_settings.get("alternatives"),
                                              columns=["mode_name", "ownership", "allowance"])
        # https://towardsdatascience.com/apply-and-lambda-usage-in-pandas-b13a1ea037f7
        mode_spec['mode_name'] = mode_spec.apply(lambda x: extract_mode_name(x['mode_name']), axis=1)
        mode_spec.sort_values('mode_name', inplace=True)
        mode_spec.drop_duplicates(subset='mode_name', inplace=True)
        return mode_spec['mode_name'].to_list()
    else:
        raise RuntimeError("Nested Logit mode structure is not expected.")