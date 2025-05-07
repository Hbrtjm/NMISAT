

def expression_1(x,y):
    return x**2 - y**2

def expression_2(x,y):
    return (x+y) * (x-y)

test_values = [
    (1e-14, 1e-14),
    (1e-14, -1e-14),
    (-1e-14, 1e-14),
    (-1e-14, -1e-14),
    (1.1e-14, 1e-14),
    (1e-14, 1.1e-14),
]

for x,y in test_values:
    expression_1_value = expression_1(x,y)
    expression_2_value = expression_2(x,y)
    print(f"Checked pair ({x},{y})")
    print(f"Value from the x^2 - y^2 {expression_1_value}")
    print(f"Value from the (x + y)(x - y) {expression_2_value}")
    print(f"Difference between expr_1 and expr_2: {expression_1_value - expression_2_value}")