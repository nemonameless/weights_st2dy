# in ppdet, only conv/bn/fc has weights
def is_conv_bn_or_fc(conv_info, bn_infos):
    if len(conv_info[1]) != 4: return 0
    match_cnt = 0
    for bn_info in bn_infos:
        if len(bn_info[1]) != 1:
            break
        if bn_info[1][0] != conv_info[1][0]:
            break
        match_cnt += 1
    if match_cnt == 1:
        return 2
    elif match_cnt == 4:
        return 1
    return 0


def check_is_conv_bn_or_fc(infos):
    is_conv_bn_or_fcs = []
    for i in range(len(infos)):
        is_conv_bn_or_fcs.append(is_conv_bn_or_fc(infos[i], infos[i+1:i+5]))
    return is_conv_bn_or_fcs


def parse_dygraph_infos(filename):
    infos = []
    with open(filename) as f:
        for line in f.readlines():
            fields = line.split()
            info = (fields[0], eval(fields[1]))
            if info not in infos:
                infos.append(info)
    return infos


def parse_static_infos(filename):
    infos = []
    with open(filename) as f:
        for line in f.readlines():
            fields = line.split()
            info = (fields[0], eval(fields[1]))
            if info not in infos:
                infos.append(info)
    return infos


def match_static_to_dygraph(static_infos, dygraph_infos, type_filename):
    match_map = {}
    st_is_conv_bn_or_fcs = check_is_conv_bn_or_fc(static_infos)
    dy_is_conv_bn_or_fcs = check_is_conv_bn_or_fc(dygraph_infos)
    st_idx = dy_idx = 0
    # for st_idx, info in enumerate(static_infos):
    with open(f"./v10_weight_name_map_{type_filename}.txt", 'w') as wf:
        while st_idx < len(static_infos):
            info = static_infos[st_idx]
            if dy_idx >= len(dygraph_infos):
                print("static weight not found in dynamic: ", static_infos[st_idx:])
                return
            if info[1] == dygraph_infos[dy_idx][1]:
                print("{:50} matched      {:50}".format(info[0], dygraph_infos[dy_idx][0]))
                wf.write("{:50} {:50}\n".format(info[0], dygraph_infos[dy_idx][0]))
                dy_idx += 1
            else:
                selects = []
                for idx in range(dy_idx + 1, len(dygraph_infos)):
                    if dygraph_infos[idx][1] == info[1]:
                        if st_is_conv_bn_or_fcs[st_idx] > 0:
                            if dy_is_conv_bn_or_fcs[idx] != st_is_conv_bn_or_fcs[st_idx]:
                                print("*****{} matched {}, but ConvBN/FC check failed, {} != {}".format(dygraph_infos[idx][0], info[0], dy_is_conv_bn_or_fcs[idx], st_is_conv_bn_or_fcs[st_idx]))
                                continue
                        selects.append(idx)
                if len(selects) > 1:
                    print("*****match wrong*******", info, dygraph_infos[dy_idx], ", is ConvBN/FC block: ", st_is_conv_bn_or_fcs[st_idx])
                    choose_str = "Please select dygraph weight name for {}:\n".format(info[0])
                    for i, idx in enumerate(selects):
                        choose_str += "\t {}. {}\n".format(i+1, dygraph_infos[idx][0])
                    choose_str += "selection: "
                    select = input(choose_str)
                    idx = selects[int(select) - 1]
                else:
                    if len(selects) == 0:
                        with open('./dy_not_match.txt', 'w') as df:
                            for idx in range(dy_idx, len(dygraph_infos)):
                                df.write("{} {}\n".format(dygraph_infos[idx][0], dygraph_infos[idx][1]))
                        with open('./st_not_match.txt', 'w') as sf:
                            for idx in range(st_idx, len(static_infos)):
                                sf.write("{} {}\n".format(static_infos[idx][0], static_infos[idx][1]))
                        print("ERROR: match wrong, not matched weight saved in dy_not_match.txt and st_not_match.txt")
                        return
                    idx = selects[0]
                dy_info = dygraph_infos.pop(idx)
                dy_is_conv_bn_or_fcs.pop(idx)
                print("{:50} matched      {:50}".format(info[0], dy_info[2]))
                wf.write("{:50} {:50}\n".format(info[0], dy_info[2]))
                if st_is_conv_bn_or_fcs[st_idx]:
                    n = 4 if st_is_conv_bn_or_fcs[st_idx] == 1 else 1
                    for i in range(n):
                        dy_info = dygraph_infos.pop(idx)
                        dy_is_conv_bn_or_fcs.pop(idx)
                        st_idx += 1
                        print("{:50} matched      {:50}".format(static_infos[st_idx][0], dy_info[2]))
                        wf.write("{:50} {:50}\n".format(static_infos[st_idx][0], dy_info[2]))

            st_idx += 1

if __name__ == "__main__":
    import sys
    dy_filename = sys.argv[1]
    st_filename = sys.argv[2]
    type_filename = sys.argv[3]
    #params, states = parse_dygraph_params_states(dy_filename)
    dygraph_infos = parse_dygraph_infos(dy_filename)
    #for info, c in dygraph_infos:
    #    print(info)
    static_infos = parse_static_infos(st_filename)
    #for info in static_infos:
    #    print(info)
    print("paddle weights number: ", len(dygraph_infos))
    print("torch weights number: ", len(static_infos))
    match_static_to_dygraph(static_infos, dygraph_infos, type_filename)

