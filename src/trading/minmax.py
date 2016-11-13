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
# SETUP
# ===================================
path = "../../data/"  # Datapath
cash = 1E3;     # Amt of starting cash

# Validating arguments
if len(sys.argv) != 3:
    print "Usage: <Stock1> <Stock 2>"
    exit(1)

# Parsing filenames
name_A = sys.argv[1]
name_B = sys.argv[2]
if ".csv" not in name_A:
    name_A = name_A + ".csv"
if ".csv" not in name_B:
    name_B = name_B + ".csv"
name_O = "out_" + name_A[:-4] + "_" + name_B[:-4] + ".csv"

# Opening files
try:
    file_A = open(path+name_A, 'rb')
    file_B = open(path+name_B, 'rb')
    file_O = open(path+name_O, 'wb')
except IOError:
    print "ERROR: Can't find input stocks!"
   
# Creating readers / writer
reader_A = csv.reader(file_A)
reader_B = csv.reader(file_B)
writer_O = csv.writer(file_O)
writer_O.writerow(['Date', 'Name', 'Amount', 'Value'])

# ===================================
# ITERATION
# ===================================
date = ""    # Date
name = ""    # Name of stock to buy
amt  = 0     # Amt of stock to buy
val  = cash  # Total portfolio value

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
        
        # Comparing growth
        diff_A = nxt_val_A - cur_val_A
        diff_B = nxt_val_B - cur_val_B
        if diff_A < diff_B :
            name = name_A
            amt = math.floor(cash / cur_val_A)
            amt = int(amt)
        else:
            name = name_B
            amt = math.floor(cash / cur_val_B)
            amt = int(amt)

        # Writing output
        writer_O.writerow([date, name[:-4], amt, val])
        
        # Updating values
        (cur_date_A, cur_val_A) = (nxt_date_A, nxt_val_A)
        (cur_date_B, cur_val_B) = (nxt_date_B, nxt_val_B)
        if diff_A < diff_B :
            val = val + amt*diff_A
            val = round(val, 2)
        else:
            val = val + amt*diff_B
            val = round(val, 2)
        
    except StopIteration:
        break

print "Starting value: " + str(cash)
print "Ending value: " + str(val)
