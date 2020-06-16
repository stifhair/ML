from matplotlib import pyplot as plt
topics = ['A', 'B', 'C', 'D', 'E']
value_a = [80, 85, 84, 83, 86]
value_b = [73, 78, 77, 82, 86]

def create_x(t, w, n, d):
    return [t*x + w*n for x in range(d)]

value_a_x = create_x(2, 1, 1, len(topics))
value_b_x = create_x(2, 1, 2, len(topics))
ax = plt.subplot()
ax.bar(value_a_x, value_a)
ax.bar(value_b_x, value_b)
middle_x = [(a+b)/2 for (a,b) in zip(value_a_x, value_b_x)]
ax.set_xticks(middle_x)
ax.set_xticklabels(topics)
plt.show()