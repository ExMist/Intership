def max_subarray(numbers):
    best_sum = 0
    best_start = best_end = 0
    current_sum = 0
    for current_end, x in enumerate(numbers):
        if current_sum <= 0:
            current_start = current_end
            current_sum = x
        else:
            current_sum += x

        if current_sum > best_sum:
            best_sum = current_sum
            best_start = current_start
            best_end = current_end + 1

    return best_sum, numbers[best_start:best_end]

if __name__ == '__main__':

    l = list(map(int,input("\nEnter the numbers : ").strip().split(',')))
    summ, subarray = max_subarray(l)
    print("Max summ: {}\nArray: {}".format(summ, subarray))