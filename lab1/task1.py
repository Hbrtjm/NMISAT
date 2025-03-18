
# l = 1
# r = -1
# mid = (l + r) / 2
# while mid / 2 != 0:
#     mid = (l + r) / 2:
#     if mid > 0:
#         l = mid
#     elif mid < 0:
#         r = mid

# print(mid)

epsilon = 1
const = 1
while const + epsilon > const:
    epsilon /= 2
epsilon *= 2
print(epsilon)