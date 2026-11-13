import math

PI = 3.14159

def power_grid_forecast(consumption_data):
    # 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
    # 2) Perform linear regression on the detrended data.
    # 3) Predict day 15's base consumption.
    # 4) Add the day 15 fluctuation back.
    # 5) Round, then add a 5% safety margin (rounded up).
    # 6) Return the final integer.
    n = len(consumption_data)
 
    cons_minus_fluct = [
        c - (10 * math.sin(2 * PI * i / 10)) 
        for i, c in enumerate(consumption_data)
    ]
    s_x_y = sum(x*y for x, y in enumerate(cons_minus_fluct, start=1))
    s_x_2 = sum(x**2 for x,_ in enumerate(cons_minus_fluct, start=1))
    s_x = sum(x for x,_y in enumerate(cons_minus_fluct, start=1))
    s_y = sum(y for _,y in enumerate(cons_minus_fluct, start=1))
    m = (n*s_x_y - s_x * s_y) / (n*s_x_2 - s_x ** 2)
    b = (s_y - m * s_x) / n
    b_15 = m * 15 + b
    pred_15 = b_15 + 10 * math.sin(2*PI * 15 / 10)
    final = math.ceil(1.05 * round(pred_15))
    return final
