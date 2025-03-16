import multiprocessing
from ortools.linear_solver import pywraplp
import json

def newSolver(name, integer=False):
    return pywraplp.Solver(name, pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING if integer else pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

def SolVal(x):
    if isinstance(x, list):
        return [SolVal(e) for e in x]
    return 0 if x is None else x if isinstance(x, (int, float)) else x.SolutionValue()

def solve_model(demands, parent_width=100):
    num_orders = len(demands)
    solver = newSolver('Cutting Stock', True)
    demands.sort(key=lambda x: -x[1])  # Sort demands by decreasing length to minimize unused widths

    k, b = bounds(demands, parent_width)
    y = [solver.IntVar(0, 1, f'y_{i}') for i in range(k[1])]
    x = [[solver.IntVar(0, b[i], f'x_{i}_{j}') for j in range(k[1])] for i in range(num_orders)]
    unused_widths = [solver.NumVar(0, parent_width, f'w_{j}') for j in range(k[1])]
    nb = solver.IntVar(k[0], k[1], 'nb')

    for i in range(num_orders):
        solver.Add(sum(x[i][j] for j in range(k[1])) >= demands[i][0])

    for j in range(k[1]):
        solver.Add(sum(demands[i][1] * x[i][j] for i in range(num_orders)) <= parent_width * y[j])
        solver.Add(parent_width * y[j] - sum(demands[i][1] * x[i][j] for i in range(num_orders)) == unused_widths[j])
        if j < k[1] - 1:
            solver.Add(sum(x[i][j] for i in range(num_orders)) >= sum(x[i][j + 1] for i in range(num_orders)))

    solver.Add(nb == solver.Sum(y[j] for j in range(k[1])))

    Cost = solver.Sum((j + 1) * y[j] for j in range(k[1]))
    solver.Minimize(Cost)

    status = solver.Solve()
    numRollsUsed = SolVal(nb)

    return status, numRollsUsed, rolls(numRollsUsed, SolVal(x), SolVal(unused_widths), demands)

def bounds(demands, parent_width=100):
    num_orders = len(demands)
    b = []
    T = 0
    k = [0, 1]
    TT = 0

    for i in range(num_orders):
        quantity, width = demands[i][0], demands[i][1]
        b.append(min(quantity, int(round(parent_width / width))))

        if T + quantity * width <= parent_width:
            T, TT = T + quantity * width, TT + quantity * width
        else:
            while quantity:
                if T + width <= parent_width:
                    T, TT, quantity = T + width, TT + width, quantity - 1
                else:
                    k[1], T = k[1] + 1, 0

    k[0] = int(round(TT / parent_width + 0.5))

    return k, b

def rolls(nb, x, w, demands):
    consumed_big_rolls = []
    num_orders = len(x)
    for j in range(len(x[0])):
        RR = [int(x[i][j]) * [demands[i][1]] for i in range(num_orders) if x[i][j] > 0]
        flattened_RR = [item for sublist in RR for item in sublist]
        waste = w[j]
        consumed_big_rolls.append((flattened_RR, waste))
    return consumed_big_rolls

def checkWidths(demands, parent_width):
    for quantity, width in demands:
        if width > parent_width:
            print(f'Small roll width {width} is greater than parent rolls width {parent_width}. Exiting')
            return False
    return True

def pair_chunks(demands, chunk_size):
    sorted_demands = sorted(demands, key=lambda x: x[1], reverse=True)
    paired_chunks = []

    while len(sorted_demands) > 0:
        first_chunk = sorted_demands[:(chunk_size // 2)]
        del sorted_demands[:(chunk_size // 2)]
        last_chunk = sorted_demands[(-chunk_size // 2):]
        del sorted_demands[(-chunk_size // 2):]

        paired_chunk = first_chunk + last_chunk
        paired_chunks.append(paired_chunk)

    return paired_chunks

def StockCutter1D(child_rolls, parent_rolls, chunk_size=8, output_json=True):
    parent_width = parent_rolls[0][1]

    if not checkWidths(demands=child_rolls, parent_width=parent_width):
        return []

    paired_chunks = pair_chunks(child_rolls, chunk_size)
    total_rolls_used = 0
    all_consumed_big_rolls = []

    for chunk in paired_chunks:
        print(f'Running Small Model for chunk {chunk}...')
        status, numRollsUsed, consumed_big_rolls = solve_model(demands=chunk, parent_width=parent_width)
        total_rolls_used += numRollsUsed
        all_consumed_big_rolls.extend(consumed_big_rolls)

    if output_json:
        with open('output.json', 'w') as outfile:
            json.dump(all_consumed_big_rolls, outfile, indent=4, sort_keys=True)

    return total_rolls_used, all_consumed_big_rolls

def backup_StockCutter1D(demands, parent_rolls):
    parent = parent_rolls[0][1]
    demands.sort(key=lambda x: -x[1])
    demands = [[demand[0], demand[1]] for demand in demands]
    print(demands)
    total_bar = []

    while any(demand[0] > 0 for demand in demands):
        bar = []
        for demand in demands:
            while demand[0] > 0 and (parent - sum(bar) >= demand[1]):
                bar.append(demand[1])
                demand[0] -= 1
        waste = parent - sum(bar)
        total_bar.append((bar, waste))

    return len(total_bar), total_bar

def find_min_total_rolls_used(demands, parent_rolls, sheet):
    chunk_sizes = [8]
    min_total_rolls_used = float('inf')
    sheet.cells(9, 10).value = "calculating"

    for chunk_size in chunk_sizes:
        sheet.cells(9, 10).value = "Finding best result.."
        print("checking value for chunk size: ", chunk_size)
        try:
            with multiprocessing.Pool(processes=1) as pool:
                result = pool.apply_async(StockCutter1D, (demands, parent_rolls, chunk_size, False))
                total_rolls_used, all_consumed_big_rolls = result.get(timeout=30)

            if total_rolls_used < min_total_rolls_used:
                min_total_rolls_used = total_rolls_used
                min_consumed_big_rolls = all_consumed_big_rolls
        except multiprocessing.TimeoutError as te:
             print("error timeout")
             continue
        
    total_rolls_used, all_consumed_big_rolls = backup_StockCutter1D(demands, parent_rolls)

    if total_rolls_used < min_total_rolls_used:
        min_total_rolls_used = total_rolls_used
        min_consumed_big_rolls = all_consumed_big_rolls

    print(f'Solved with total rolls used: {min_total_rolls_used}.')
    print('Cuts for each roll:')

    for i, (roll, waste) in enumerate(min_consumed_big_rolls, 1):
        print(f'Roll {i}: {roll}, Waste: {waste:.2f}')

    return min_total_rolls_used, min_consumed_big_rolls

import xlwings as xw


data_list = []

def get_max_row(startRow, startCol, sheet):
    while sheet.cells(startRow, startCol).value != None:
        startRow += 1
    return startRow

def get_demand(sheet):
    print("i got called")
    for i in range(12, get_max_row(12,7, sheet)):
        bar_length = int(sheet.cells(i, 7).value)
        bar_quantity = int(sheet.cells(i, 8).value)
        
        print(bar_length, bar_quantity)
        print(i)
        sheet.cells(9,10).value = "Getting Data.."
        
        # Check if bar_length is already present in the data_list
        found = False
        for index, (quantity, length) in enumerate(data_list):
            if length == bar_length:
                # If found, update the quantity by adding the current quantity
                data_list[index] = (quantity + bar_quantity, length)
                found = True
                break
        
        # If not found, add the new pair to the data_list
        if not found:
            data_list.append((bar_quantity, bar_length))
    
    # Sort the data_list based on bar_length in descending order
    data_list.sort(key=lambda x: x[1], reverse=True)
    print(data_list)
    return data_list

def main():
    wb = xw.Book.caller()
    sheet = wb.sheets[0]

    demands = get_demand(sheet)
    for i, (quantity, length) in enumerate(demands, 1):
        sheet.cells(11+i,9).value = (length)
        sheet.cells(11+i,10).value = (quantity)
    
    parent_length = sheet.cells(9,8).value
    parent_rolls = [(0, parent_length)]
    minNumRollsUsed, all_consumed_big_rolls = find_min_total_rolls_used(demands, parent_rolls, sheet)
    sheet.cells(9,10).value = minNumRollsUsed
    roll_counter = 1
    for i, (roll, waste) in enumerate(all_consumed_big_rolls, 1):
        if roll:
            sheet.cells(11+roll_counter,9).value = (f'Bar {roll_counter}:')
            sheet.cells(11+roll_counter,10).value = (f'{roll}, Waste: {waste:.2f}')
            roll_counter += 1


if __name__ == "__main__":
    xw.Book("StockCutSolver.xlsm").set_mock_caller()
    main()


# def main():
#     wb = xw.Book.caller()
#     sheet = wb.sheets[0]
#     if sheet["A1"].value == "Hello xlwings!":
#         sheet["A1"].value = "Bye xlwings!"
#     else:
#         sheet["A1"].value = "Hello xlwings!"


# @xw.func
# def hello(name):
#     return f"Hello {name}!"

