def gini_index(splitted_grps, all_classes):
    result = 0
    total_size = sum([len(i) for i in splitted_grps])
    for grp in splitted_grps:
        grp_prob = 0
        for i in all_classes:
            if not len(grp):
                continue
            class_prob = [row[-1] for row in grp].count(i) / len(grp)
            grp_prob += class_prob**2
        result += (1 - grp_prob) * (len(grp) / total_size)
    return float("{:.2f}".format(result))