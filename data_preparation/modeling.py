
AGE_MAP = [
    (0, None),
    (18, 0), # kids
    (60, 1), # adults
    # above 60 is seniors
]

CABIN_MAP = {
    'G':0,
    'F':1,
    'E':2,
    'D':3,
    'C':4,
    'B':5,
    'A':6
}

def data2vec(object):
    """
        This function will return a numpy vector of values in the following order:
            [Pclass, Sex, Age, Horizontal_family, Vertical_family, Cabin]
    """

    data = []
    #ignore ID and result

    data.append(int(row[2]))

    #ignore Name since that is doubtful to matter

    data.append(['male', 'female'].index(row[4]))

    age_years = int(row[5])
    # pick out the age class
    age = 2
    for k, v in AGE_MAP.items():
        if age_years < k:
            age = v
            break
    data.append(age)

    data.append(int(row[6]))
    data.append(int(row[7]))
    #ignore ticket number
    #ignore fare

    # take the first letter of the cabin
    # could indicate a location that may be less advantageous when the crash occured
    data.append(CABIN_MAP.get((row[10] or 'G')[0], 0))
