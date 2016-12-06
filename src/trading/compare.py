import sys, csv

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

# Validating arguments
if len(sys.argv) != 5:
    print "Usage: <Portfolio 1> <Portfolio 2> <Stock 1> <Stock 2>"
    exit(1)

# Parsing filenames, creating paths
sys.argv = [word.replace('.csv', '') for word in sys.argv]
name_A = sys.argv[1]
name_B = sys.argv[2]
port_name = sys.argv[3] + "_" + sys.argv[4] + ".csv"

path_A = path + name_A + "/" + port_name
path_B = path + name_B + "/" + port_name
path_O = path + "comp/" + name_A + "_" + name_B + "_" + port_name
print path_A
print path_B
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
headers = ['Date', name_A+' Total', name_B+' Total', 'Diff.', 'Total Diff.']
writer_O.writerow(headers)

# ===================================
# Iteration
# ===================================
num = 0
den = 0

# Skipping headers
reader_A.next()
reader_B.next()

while True:
    try:
        (date_A, _, _, _, _, _, total_A) = reader_A.next()
        (date_B, _, _, _, _, _, total_B) = reader_B.next()
        total_A = float(total_A)
        total_B = float(total_B)
        
        check_dates(date_A, name_A, date_B, name_B)
        
        diff = abs((total_A - total_B))
        num = num + diff
        den = den + total_B
        diff = round(diff / total_B, 3)
        total_diff = round(num / den, 3)
        
        row = [date_A, total_A, total_B, diff, total_diff]
        writer_O.writerow(row)
    
    except StopIteration:
        break