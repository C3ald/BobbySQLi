from sqlmap import tamper

attributes = dir(tamper)

funcs = [getattr(tamper, attr) for attr in attributes if callable(getattr(tamper, attr))]

for func in funcs:
        result = func()
        print(result)