import importlib
from itertools import combinations
from functools import partial
# Update the list of available tampers to only include those that have the 'keywords' attribute
available_tampers = ['apostrophemask', 'apostrophenullencode', 'appendnullbyte', 'base64encode', 'between', 'bluecoat', 'chardoubleencode',
                     'charencode', 'charunicodeencode', 'concat2concatws', 'equaltolike', 'greatest', 'ifnull2ifisnull',
                     'least', 'lowercase', 'modsecurityzeroversioned', 'multiplespaces', 'overlongutf8', 'percentage', 'randomcase', 'randomcomments', 'space2comment', 'space2hash',
                     'space2morehash', 'space2mysqlblank', 'space2mysqldash', 'space2plus', 'space2randomblank', 'sp_password', 'unionalltounion',
                     'unmagicquotes', 'uppercase']

tampers = []

# Import the tamper modules and add them to the tampers list
for tamper in available_tampers:
    module = importlib.import_module('sqlmap.tamper.' + tamper)
    tampers.append(module.tamper)

# print(len(tampers))


def get_tampered_payloads(payload, level=1):
    final_payloads = []
    level = len(tampers) - level
    final_payloads.append(payload)
    write_payload_to_file(payload, mode='w')
    for i in range(len(available_tampers) - level):
        combs = get_combos(i)
        combs = list(set(combs))
        for comb in combs:
            comb = list(set(comb))
            for c in comb:
                # Call the tamper method of the module to get the tampered payload
                try:
                    tamper_func = c(payload)
                    pay = tamper_func
            # If the tamper method returns a string, then the tamper)
            # final_payloads.append(pay)
                    if pay not in final_payloads:
                        write_payload_to_file(pay)
                        final_payloads.append(pay)
                except AttributeError:
                    None
        print(i)
    return 1


def get_combos(combo_num):
    comb = list(combinations(tampers, combo_num))
    return comb


def write_payload_to_file(payload, mode='a'):
    with open('tampered_payload_data.txt', mode) as f:
        f.write(str(payload) + '\n')
        if mode != 'a':
            f.close()


if __name__ == '__main__':
    payload = "1' OR 1=1--"
    tampered_payloads = get_tampered_payloads(payload)
    print(tampered_payloads)
