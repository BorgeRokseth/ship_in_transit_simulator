# Python3 code to demonstrate
# to find index of first element just
# greater than K
# using map() + index()

# initializing list
test_list = [0.4, 0.5, 11.2, 8.4, 10.4]

# printing original list
print("The original list is : " + str(test_list))

# using map() + index()
# to find index of first element just
# greater than 0.6
res = list(map(lambda i: i > 0.6, test_list)).index(True)

# printing result
print("The index of element just greater than 0.6 : "
      + str(res))