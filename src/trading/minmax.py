import sys, csv, math

# ===================================
# FUNCTION: Checking dates
# ===================================  
def check_dates(date_A, name_A, date_B, name_B):
    if date_A != date_B:
        print "ERROR: Mismatching dates, " + \
            date_A + "(" + name_A + ") vs " + \
            date_B + "(" + name_B + ")"
        exit(1)
    return

# ===================================
# FUNCTION: Perform min/max, write outputs
# ===================================
def minmax(name_A, name_B, opt, \
   in_path="../../data/", out_path="../../data/", init=1000):
    
    # ===================================
    # SETUP
    # ===================================
    
    # Parsing filenames, creating paths
    path_A = in_path + name_A
    if ".csv" not in path_A:
        path_A = path_A + ".csv"

    path_B = in_path + name_B
    if ".csv" not in path_B:
        path_B = path_B + ".csv"

    path_O = out_path + name_A + "_" + name_B + ".csv"
    name_A = name_A.upper()
    name_B = name_B.upper()

    opt = False if opt.lower() == "min" else True

    # Opening files
    try:
        file_A = open(path_A, 'rb')
        file_B = open(path_B, 'rb')
        file_O = open(path_O, 'wb')
    except IOError:
        print "ERROR: Can't find input stocks!"
        exit(1)
       
    # Creating readers / writer
    reader_A = csv.reader(file_A)
    reader_B = csv.reader(file_B)
    writer_O = csv.writer(file_O)
    headers = ['Date', name_A+' Val', name_A+' Amt', name_B+' Val', name_B+' Amt', 'Cash', 'Total']
    writer_O.writerow(headers)

    # ===================================
    # ITERATION
    # ===================================
    date    = ""    # Date
    amt_A   = 0     # Amt of stock A to buy
    amt_B   = 0     # Amt of stock B to buy
    cash    = 0     # Amt of leftover cash
    total   = init  # Amt of total value

    # Getting current stocks
    (cur_date_A, cur_val_A) = reader_A.next()
    (cur_date_B, cur_val_B) = reader_B.next()
    cur_val_A = float(cur_val_A)
    cur_val_B = float(cur_val_B)

    check_dates(cur_date_A, name_A, cur_date_B, name_B)

    while True:
        date = cur_date_A

        try:
            # Getting next stocks
            (nxt_date_A, nxt_val_A) = reader_A.next()
            (nxt_date_B, nxt_val_B) = reader_B.next()
            check_dates(nxt_date_A, name_A, nxt_date_B, name_B)
            nxt_val_A = float(nxt_val_A)
            nxt_val_B = float(nxt_val_B)
            amt_A = 0
            amt_B = 0

            # Comparing growth
            diff_A = nxt_val_A - cur_val_A
            amt_A  = math.floor(total / cur_val_A)
            amt_A  = int(amt_A)
            diff_A = amt_A * diff_A

            diff_B = nxt_val_B - cur_val_B
            amt_B  = math.floor(total / cur_val_B)
            amt_B  = int(amt_B)
            diff_B = amt_B * diff_B

            if opt != (diff_A < diff_B) :
                amt_B = 0
            else:
                amt_A = 0

            # Writing output
            cash = total - (amt_A*cur_val_A + amt_B*cur_val_B)
            cash = round(cash, 2)
            row = [date, cur_val_A, amt_A, cur_val_B, amt_B, cash, total]
            writer_O.writerow(row)
            
            # Updating values
            (cur_date_A, cur_val_A) = (nxt_date_A, nxt_val_A)
            (cur_date_B, cur_val_B) = (nxt_date_B, nxt_val_B)
            if opt != (diff_A < diff_B) :
                total = total + diff_A
                total = round(total, 2)
            else:
                total = total + diff_B
                total = round(total, 2)
            
        except StopIteration:
            break

# ===================================
# MAIN
# ===================================

# Validating arguments
if len(sys.argv) != 3:
    print "Usage: <Stock1> <Stock 2>"
    exit(1)
    
name_A = sys.argv[1]
name_B = sys.argv[2]
    
for opt in ['min', 'max']:
    in_dir  = "../../data/"
    out_dir = "../../data/bench-" + opt + "/"
    minmax(name_A, name_B, opt, in_dir, out_dir)