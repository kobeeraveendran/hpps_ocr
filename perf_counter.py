with open("logs/ocr.log") as f:
    accepted = f.read().count(">")
    f.seek(0, 0)
    for i, l in enumerate(f):
        pass
    
    total = i - 1
print("Num. acceptable: {}/{}".format(accepted, total))